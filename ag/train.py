#!/usr/bin/env python3
import argparse, random, os, glob, time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# Import conversion functions from pyg.py
from pyg import load_master_topology, build_plain_sample, NODE_FEATURE_KEYS, ADD_IS_ACTIVE_FLAG

class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Ignore edge_attr - GCNConv cannot handle negative edge weights
        x = F.relu(self.conv1(x, edge_index, edge_weight=None))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=None))
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.lin(x), dim=-1)

INJECTED_VARIANTS = ["-nve.json", "-cmp.json", "-esc.json", "-ign.json"]

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
    """Load one benign and one random injected variant for a given file ID."""
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
    benign_data = build_plain_sample(benign_path, id2idx, label=0,
                                     node_feature_keys=NODE_FEATURE_KEYS,
                                     add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                     use_float16=use_float16)
    benign_graph = Data(**benign_data)
    benign_size_mb = sum(v.element_size() * v.nelement() for v in [benign_graph.x, benign_graph.edge_index, benign_graph.edge_attr]) / (1024**2)
    # print(f"[Graph] Benign {file_id}: {benign_size_mb:.2f} MB (nodes={benign_graph.x.size(0)}, edges={benign_graph.edge_index.size(1)})")

    injected_data = build_plain_sample(injected_path, id2idx, label=1,
                                       node_feature_keys=NODE_FEATURE_KEYS,
                                       add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                       use_float16=use_float16)
    injected_graph = Data(**injected_data)
    injected_size_mb = sum(v.element_size() * v.nelement() for v in [injected_graph.x, injected_graph.edge_index, injected_graph.edge_attr]) / (1024**2)
    # print(f"[Graph] Injected {file_id}: {injected_size_mb:.2f} MB (nodes={injected_graph.x.size(0)}, edges={injected_graph.edge_index.size(1)})")

    return benign_graph, injected_graph

def sample_batch(data_dir, file_ids, batch_size, id2idx, device, use_float16=True):
    """Sample a balanced batch: batch_size benign + batch_size injected."""
    sampled_ids = random.sample(file_ids, min(batch_size, len(file_ids)))

    data_list = []
    for file_id in sampled_ids:
        benign, injected = load_graph_pair(data_dir, file_id, id2idx, use_float16=use_float16)
        data_list.append(benign)
        data_list.append(injected)

    return Batch.from_data_list(data_list).to(device)

def run_epoch(model, data_dir, file_ids, batch_size, id2idx, optimizer=None, device="cpu", num_batches=None, use_float16=True):
    """Run one epoch by sampling batches on-the-fly."""
    model.train(optimizer is not None)

    # Calculate number of batches
    if num_batches is None:
        num_batches = max(1, len(file_ids) // batch_size)

    total = correct = 0
    loss_sum = 0.0

    for i in range(num_batches):
        batch = sample_batch(data_dir, file_ids, batch_size, id2idx, device, use_float16=use_float16)

        if optimizer:
            optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        loss = F.nll_loss(out, batch.y.view(-1))

        if optimizer:
            loss.backward()
            optimizer.step()

        pred = out.argmax(dim=-1)
        total += batch.y.size(0)
        correct += int((pred == batch.y.view(-1)).sum())
        loss_sum += float(loss) * batch.y.size(0)
        print(f"--- batch {i} completed")
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
    ap.add_argument("--output", type=str, default="trained_gcn.pt")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--float16", action="store_true", help="Use float16 for node/edge features to save memory")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    print(f"[+] Using float16: {args.float16}")
    print(f"[+] Batch size: {args.batch_size} IDs ({args.batch_size * 2} graphs per batch)")

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

    # Initialize model
    model = GCNGraphClassifier(in_dim=in_dim, hidden=args.hidden, num_classes=num_classes).to(device)

    # Convert model to float16 if requested
    if args.float16:
        model = model.half()
        print(f"[+] Model converted to float16")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, args.data_dir, train_ids, args.batch_size, id2idx,
                                     optimizer=opt, device=device, use_float16=args.float16)
        va_loss, va_acc = run_epoch(model, args.data_dir, val_ids, args.batch_size, id2idx,
                                     optimizer=None, device=device, use_float16=args.float16)
        print(f"epoch {epoch:03d} [{time.time() - start_time}s] | train {tr_acc:.3f} loss {tr_loss:.4f} | val {va_acc:.3f} loss {va_loss:.4f}")

    # Test evaluation
    te_loss, te_acc = run_epoch(model, args.data_dir, test_ids, args.batch_size, id2idx,
                                 optimizer=None, device=device, use_float16=args.float16)
    print(f"[test] acc {te_acc:.3f} loss {te_loss:.4f}")

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "num_classes": num_classes,
        "hidden": args.hidden
    }, args.output)
    print(f"[✓] Saved trained model to {args.output}")

if __name__ == "__main__":
    main()
