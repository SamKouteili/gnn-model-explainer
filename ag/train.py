#!/usr/bin/env python3
import argparse, random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

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

def split_dataset(graphs, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.Random(seed).shuffle(graphs)
    n = len(graphs); n_train = int(n*train_ratio); n_val = int(n*val_ratio)
    return graphs[:n_train], graphs[n_train:n_train+n_val], graphs[n_train+n_val:]

def run_epoch(model, loader, optimizer=None, device="cpu"):
    model.train(optimizer is not None)
    total=correct=0; loss_sum=0.0
    for data in loader:
        data = data.to(device)
        if optimizer: optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        loss = F.nll_loss(out, data.y.view(-1))
        if optimizer:
            loss.backward(); optimizer.step()
        pred = out.argmax(dim=-1)
        total += data.y.size(0)
        correct += int((pred == data.y.view(-1)).sum())
        loss_sum += float(loss) * data.y.size(0)
    return loss_sum/max(total,1), correct/max(total,1)

def main():
    ap = argparse.ArgumentParser(description="Train GCN on AG data.pt")
    ap.add_argument("bundle_path", type=str)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--output", type=str, default="trained_gcn.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load plain dicts and reconstruct Data objects
    bundle = torch.load(args.bundle_path, map_location="cpu", weights_only=False)
    graphs_plain = bundle["graphs"]
    graphs = [Data(**g) for g in graphs_plain]
    in_dim = graphs[0].x.size(1)
    num_classes = int(max(d.y.max().item() for d in graphs) + 1)

    train_set, val_set, test_set = split_dataset(graphs, 0.8, 0.1, seed=42)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    model = GCNGraphClassifier(in_dim=in_dim, hidden=args.hidden, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer=opt, device=device)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer=None, device=device)
        print(f"epoch {epoch:03d} | train {tr_acc:.3f} loss {tr_loss:.4f} | val {va_acc:.3f} loss {va_loss:.4f}")

    te_loss, te_acc = run_epoch(model, test_loader, optimizer=None, device=device)
    print(f"[test] acc {te_acc:.3f} loss {te_loss:.4f}")

    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "num_classes": num_classes, "hidden": args.hidden},
               args.output)
    print(f"[✓] Saved trained model to {args.output}")

if __name__ == "__main__":
    main()
