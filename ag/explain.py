#!/usr/bin/env python3
import argparse, os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

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

def main():
    ap = argparse.ArgumentParser(description="Explain GCN predictions using GNNExplainer")
    ap.add_argument("model_path", type=str, help="Path to trained model .pt file")
    ap.add_argument("bundle_path", type=str, help="Path to data bundle")
    ap.add_argument("--graph-idx", type=int, default=0, help="Index of graph to explain")
    ap.add_argument("--explainer-epochs", type=int, default=500)
    ap.add_argument("--output-dir", type=str, default="explanations")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    in_dim = checkpoint["in_dim"]
    num_classes = checkpoint["num_classes"]
    hidden = checkpoint.get("hidden", 128)

    model = GCNGraphClassifier(in_dim=in_dim, hidden=hidden, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[✓] Loaded model from {args.model_path}")

    # Load data bundle
    bundle = torch.load(args.bundle_path, map_location="cpu", weights_only=False)
    graphs_plain = bundle["graphs"]
    graphs = [Data(**g) for g in graphs_plain]

    if args.graph_idx >= len(graphs):
        print(f"[!] Graph index {args.graph_idx} out of range (max: {len(graphs)-1})")
        return

    to_explain = graphs[args.graph_idx].to(device)
    print(f"[✓] Loaded graph {args.graph_idx} with {to_explain.num_nodes} nodes and {to_explain.num_edges} edges")

    # Create explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=args.explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="log_probs",
        ),
    )

    # Generate explanation
    print(f"[...] Generating explanation (this may take a while)...")
    explanation = explainer(
        to_explain.x,
        to_explain.edge_index,
        edge_attr=to_explain.edge_attr,
        batch=torch.zeros(to_explain.num_nodes, dtype=torch.long, device=to_explain.x.device),
        index=0,
    )
    print(f"[✓] Generated explanations: {explanation.available_explanations}")

    # Save visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        feat_path = os.path.join(args.output_dir, "feature_importance.png")
        explanation.visualize_feature_importance(feat_path, top_k=10)
        print(f"[✓] Saved feature importance to {feat_path}")
    except Exception as e:
        print(f"[!] Feature importance visualization failed: {e}")

    try:
        graph_path = os.path.join(args.output_dir, "subgraph.pdf")
        explanation.visualize_graph(graph_path)
        print(f"[✓] Saved subgraph visualization to {graph_path}")
    except Exception as e:
        print(f"[!] Subgraph visualization failed: {e}")

if __name__ == "__main__":
    main()
