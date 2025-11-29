#!/usr/bin/env python3
import argparse, random, os, glob, time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import conversion functions from pyg.py
from pyg import load_master_topology, build_plain_sample, NODE_FEATURE_KEYS, ADD_IS_ACTIVE_FLAG

class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Use absolute value of edge weights (GCNConv cannot handle negative weights)
        edge_weight = None
        if edge_attr is not None:
            edge_weight = edge_attr.abs().squeeze(-1)  # [E, 1] -> [E] with abs values

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.lin(x), dim=-1)

INJECTED_VARIANTS = ["-nve.json", "-cmp.json", "-esc.json", "-ign.json"]

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the directory.

    Returns:
        tuple: (checkpoint_path, start_epoch, history) or (None, 1, {}) if no checkpoint found
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "gcn*.pt"))
    if not checkpoint_files:
        return None, 1, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Extract epoch numbers and find the latest
    checkpoints = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        if basename.startswith("gcn") and basename.endswith(".pt"):
            try:
                # Extract epoch number from gcnXXX.pt
                epoch_str = basename[3:-3]  # Remove "gcn" and ".pt"
                epoch = int(epoch_str)
                checkpoints.append((epoch, path))
            except ValueError:
                continue

    if not checkpoints:
        return None, 1, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Get the latest checkpoint
    latest_epoch, latest_path = max(checkpoints, key=lambda x: x[0])
    print(f"[+] Found checkpoint: {latest_path} (epoch {latest_epoch})")

    checkpoint = torch.load(latest_path)
    history = checkpoint.get("history", {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})

    return latest_path, latest_epoch + 1, history

def discover_file_ids(data_dir):
    """Discover all base file IDs (e.g., '0', '1', '2') from benign directory."""
    benign_dir = os.path.join(data_dir, "benign")
    benign_files = glob.glob(os.path.join(benign_dir, "*.json"))

    # Extract base IDs (e.g., '0' from '0.json')
    file_ids = set()
    for path in benign_files:
        basename = os.path.basename(path)
        base_id = basename.replace(".json", "")
        file_ids.add(base_id)

    return sorted(file_ids)

def split_ids(file_ids, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split file IDs into train/val/test sets."""
    ids = list(file_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids

def load_graph_pair(data_dir, file_id, id2idx, use_float16=True):
    """Load one benign and one random injected variant for a given file ID.

    Returns:
        Tuple of (benign_graph, injected_graph) or None if files are corrupted.
    """
    benign_path = os.path.join(data_dir, "benign", f"{file_id}.json")

    # Try to find injected variant (with fallback for data_small format)
    injected_path = None
    for variant in INJECTED_VARIANTS:
        candidate = os.path.join(data_dir, "injected", f"{file_id}{variant}")
        if os.path.exists(candidate):
            injected_path = candidate
            break

    # Fallback: if no variant found, try plain .json
    if injected_path is None:
        candidate = os.path.join(data_dir, "injected", f"{file_id}.json")
        if os.path.exists(candidate):
            injected_path = candidate

    # If still not found, randomly pick one variant for main dataset
    if injected_path is None:
        variant = random.choice(INJECTED_VARIANTS)
        injected_path = os.path.join(data_dir, "injected", f"{file_id}{variant}")

    # Convert both to PyG Data objects with float16 for memory savings
    try:
        benign_data = build_plain_sample(benign_path, id2idx, label=0,
                                         node_feature_keys=NODE_FEATURE_KEYS,
                                         add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                         use_float16=use_float16)
        benign_graph = Data(**benign_data)
    except Exception as e:
        print(f"[!] CORRUPTED FILE (benign): {benign_path}")
        print(f"[!] Error: {e}")
        return None

    try:
        injected_data = build_plain_sample(injected_path, id2idx, label=1,
                                           node_feature_keys=NODE_FEATURE_KEYS,
                                           add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                           use_float16=use_float16)
        injected_graph = Data(**injected_data)
    except Exception as e:
        print(f"[!] CORRUPTED FILE (injected): {injected_path}")
        print(f"[!] Error: {e}")
        return None

    return benign_graph, injected_graph

def sample_batch(data_dir, file_ids, batch_size, id2idx, device, use_float16=True, max_retries=10):
    """Sample a balanced batch: batch_size benign + batch_size injected.

    Handles corrupted files by retrying with different samples.
    """
    data_list = []
    attempts = 0
    available_ids = list(file_ids)

    while len(data_list) < batch_size * 2 and attempts < max_retries and available_ids:
        # Sample one file ID
        file_id = random.choice(available_ids)
        available_ids.remove(file_id)

        result = load_graph_pair(data_dir, file_id, id2idx, use_float16=use_float16)

        if result is not None:
            benign, injected = result
            data_list.append(benign)
            data_list.append(injected)
        else:
            # Corrupted file, will try another
            attempts += 1

    if len(data_list) == 0:
        raise RuntimeError(f"Failed to load any valid graphs after {max_retries} attempts")

    return Batch.from_data_list(data_list).to(device)

def run_epoch(model, data_dir, file_ids, batch_size, id2idx, optimizer=None, device="cpu", num_batches=None, use_float16=True, max_num_files=None, scaler=None):
    """Run one epoch by sampling batches on-the-fly."""
    model.train(optimizer is not None)

    # Calculate number of batches
    if num_batches is None:
        effective_files = len(file_ids)
        if max_num_files is not None and max_num_files < effective_files:
            effective_files = max_num_files
        num_batches = max(1, effective_files // batch_size)

    total = correct = 0
    loss_sum = 0.0

    for i in range(num_batches):
        batch = sample_batch(data_dir, file_ids, batch_size, id2idx, device, use_float16=use_float16)

        if optimizer:
            optimizer.zero_grad()

        # Use autocast for mixed precision
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=(scaler is not None)):
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            loss = F.nll_loss(out, batch.y.view(-1))

        # Check for NaN in output
        if torch.isnan(out).any():
            print(f"[!] NaN detected in model output at batch {i}")
            print(f"[!] Input x has NaN: {torch.isnan(batch.x).any()}")
            print(f"[!] Output stats: min={out.min()}, max={out.max()}, mean={out.mean()}")

        if optimizer:
            if scaler is not None:
                # Scaled backprop for mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm):
                    print(f"[NaN GradNorm] batch {i} grad_norm: {grad_norm:.4f}")
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if i == 0 or torch.isnan(grad_norm):
                    print(f"[NaN GradNorm] batch {i} grad_norm: {grad_norm:.4f}")
                optimizer.step()

        pred = out.argmax(dim=-1)
        total += batch.y.size(0)
        correct += int((pred == batch.y.view(-1)).sum())
        loss_sum += float(loss) * batch.y.size(0)

        if i == 0:  # Print first batch details
            print(f"--- batch {i} completed | loss: {float(loss):.4f}, acc: {int((pred == batch.y.view(-1)).sum())}/{batch.y.size(0)}")
    
    return loss_sum / max(total, 1), correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser(description="Train GCN on attribution graphs (on-the-fly loading)")
    ap.add_argument("data_dir", type=str, help="Directory containing benign/ and injected/ folders")
    ap.add_argument("--vocabulary", type=str, default=None, help="Path to vocabulary.json (default: <data_dir>/vocabulary.json)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4, help="Number of IDs to sample per batch (actual batch will be 2x this)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--out", type=str, default="models")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--float16", action="store_true", help="Use float16 for node/edge features to save memory")
    ap.add_argument("--max-num-files", type=int, default=None, help="Maximum number of file IDs to use per epoch (actual graphs will be 2x this)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    print(f"[+] Using float16: {args.float16}")
    print(f"[+] Batch size: {args.batch_size} IDs ({args.batch_size * 2} graphs per batch)")
    if args.max_num_files is not None:
        print(f"[+] Max files per epoch: {args.max_num_files} ({args.max_num_files * 2} graphs)")

    # Load vocabulary
    vocab_path = args.vocabulary or os.path.join(args.data_dir, "vocabulary.json")
    if not os.path.exists(vocab_path):
        print(f"[!] Vocabulary not found at {vocab_path}")
        print(f"[!] Please run: python vocabulary.py {args.data_dir}")
        return

    id2idx, idx2id = load_master_topology(vocab_path)
    print(f"[+] Loaded vocabulary: {len(idx2id)} nodes")

    # Discover and split file IDs
    file_ids = discover_file_ids(args.data_dir)
    print(f"[+] Found {len(file_ids)} file IDs")

    train_ids, val_ids, test_ids = split_ids(file_ids, args.train_ratio, args.val_ratio, args.seed)
    print(f"[+] Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Determine feature dimensionality from first sample
    benign, injected = load_graph_pair(args.data_dir, train_ids[0], id2idx, use_float16=args.float16)
    in_dim = benign.x.size(1)
    num_classes = 2
    print(f"[+] Feature dim: {in_dim}, Num classes: {num_classes}")
    print(f"[+] Nodes per graph: {benign.x.size(0)}, Edges: {benign.edge_index.size(1)}")

    # Initialize model (keep in float32)
    model = GCNGraphClassifier(in_dim=in_dim, hidden=args.hidden, num_classes=num_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Use GradScaler for float16 training stability
    scaler = GradScaler() if args.float16 else None
    if args.float16:
        print(f"[+] Using mixed precision training with GradScaler")

    # Check for existing checkpoints
    checkpoint_path, start_epoch, history = find_latest_checkpoint(args.out)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        opt.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"[+] Resumed from epoch {start_epoch - 1}")
    else:
        print(f"[+] Starting training from scratch")

    # Calculate checkpoint epochs (1/4, 2/4, 3/4)
    checkpoint_epochs = {
        args.epochs // 4,
        args.epochs // 2,
        (3 * args.epochs) // 4
    }

    # Training loop
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, args.data_dir, train_ids, args.batch_size, id2idx,
                                     optimizer=opt, device=device, use_float16=args.float16,
                                     max_num_files=args.max_num_files, scaler=scaler)
        va_loss, va_acc = run_epoch(model, args.data_dir, val_ids, args.batch_size, id2idx,
                                     optimizer=None, device=device, use_float16=args.float16,
                                     max_num_files=args.max_num_files, scaler=scaler)
        print(f"[epoch] {epoch:03d} [{time.time() - start_time}s] | train {tr_acc:.3f} loss {tr_loss:.4f} | val {va_acc:.3f} loss {va_loss:.4f}")

        # Save history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Save checkpoint at 1/4, 2/4, 3/4 epochs
        if epoch in checkpoint_epochs:
            checkpoint_data = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "history": history,
                "in_dim": in_dim,
                "num_classes": num_classes,
                "hidden": args.hidden
            }
            if scaler is not None:
                checkpoint_data["scaler"] = scaler.state_dict()

            checkpoint_file = os.path.join(args.out, f"gcn{epoch:03d}.pt")
            torch.save(checkpoint_data, checkpoint_file)
            print(f"[+] Saved checkpoint: {os.path.basename(checkpoint_file)}")

    # Test evaluation
    te_loss, te_acc = run_epoch(model, args.data_dir, test_ids, args.batch_size, id2idx,
                                 optimizer=None, device=device, use_float16=args.float16,
                                 max_num_files=args.max_num_files, scaler=scaler)
    print(f"[test] acc {te_acc:.3f} loss {te_loss:.4f}")

    # Save final model
    final_path = os.path.join(args.out, "ag_gcn.pt")
    final_checkpoint = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "history": history,
        "in_dim": in_dim,
        "num_classes": num_classes,
        "hidden": args.hidden,
        "test_loss": te_loss,
        "test_acc": te_acc
    }
    if scaler is not None:
        final_checkpoint["scaler"] = scaler.state_dict()
    torch.save(final_checkpoint, final_path)
    print(f"[✓] Saved final model to {final_path}")

    # Plot training history
    if len(history["train_loss"]) > 0:
        plot_path = os.path.join(args.out, "training_loss.png")
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs_range, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs_range, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Over Time', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"[✓] Saved loss plot to {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
