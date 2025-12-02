#!/usr/bin/env python3
import argparse, os, glob
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

# Import from your existing code
from pyg import load_master_topology, build_plain_sample, NODE_FEATURE_KEYS, ADD_IS_ACTIVE_FLAG

class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Create variable number of GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.lin = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Convert float16 input to float32 for computation
        if x.dtype == torch.float16:
            x = x.float()

        # Use absolute value of edge weights (GCNConv cannot handle negative weights)
        edge_weight = None
        if edge_attr is not None:
            edge_weight = edge_attr.abs().squeeze(-1)
            if edge_weight.dtype == torch.float16:
                edge_weight = edge_weight.float()

        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return F.log_softmax(self.lin(x), dim=-1)

def main():
    ap = argparse.ArgumentParser(description="Explain GCN predictions using GNNExplainer")
    ap.add_argument("model_path", type=str, help="Path to trained model .pt file")
    ap.add_argument("data_dir", type=str, help="Path to data directory containing benign/ and injected/")
    ap.add_argument("--graph-file", type=str, default=None, help="Specific JSON file to explain (e.g., benign/0.json)")
    ap.add_argument("--vocabulary", type=str, default=None, help="Path to vocabulary.json")
    ap.add_argument("--explainer-epochs", type=int, default=100)
    ap.add_argument("--output-dir", type=str, default="explanations")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    in_dim = checkpoint["in_dim"]
    num_classes = checkpoint["num_classes"]
    hidden = checkpoint.get("hidden", 128)
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.5)

    model = GCNGraphClassifier(in_dim=in_dim, hidden=hidden, num_classes=num_classes,
                                num_layers=num_layers, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[✓] Loaded model from {args.model_path}")

    # Load vocabulary
    vocab_path = args.vocabulary or os.path.join(args.data_dir, "vocabulary.json")
    id2idx, idx2id = load_master_topology(vocab_path)
    print(f"[✓] Loaded vocabulary: {len(idx2id)} nodes")

    # Load graph to explain
    if args.graph_file:
        graph_path = os.path.join(args.data_dir, args.graph_file)
    else:
        # Default to first benign graph
        graph_path = os.path.join(args.data_dir, "benign/0.json")

    if not os.path.exists(graph_path):
        print(f"[!] Graph file not found: {graph_path}")
        return

    # Determine label from path
    label = 1 if "injected" in graph_path else 0
    graph_dict = build_plain_sample(graph_path, id2idx, label=label,
                                     node_feature_keys=NODE_FEATURE_KEYS,
                                     add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                     use_float16=False)  # Use float32 for explanation
    to_explain = Data(**graph_dict).to(device)
    print(f"[✓] Loaded graph from {graph_path}")
    print(f"    Nodes: {to_explain.num_nodes}, Edges: {to_explain.num_edges}, Label: {label}")

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
