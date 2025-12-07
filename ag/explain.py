#!/usr/bin/env python3
import argparse, os, glob
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from collections import defaultdict
import numpy as np

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

def explain_single_graph(model, explainer, graph_data, idx2id, device):
    """
    Explain a single graph and return node/edge importance scores.
    Returns: (node_mask, edge_mask, active_mask, edge_index, edge_attr, prediction)
    """
    explanation = explainer(
        graph_data.x,
        graph_data.edge_index,
        edge_attr=graph_data.edge_attr,
        batch=torch.zeros(graph_data.num_nodes, dtype=torch.long, device=graph_data.x.device),
        index=0,
    )

    node_mask = explanation.node_mask.cpu().numpy().squeeze()
    edge_mask = explanation.edge_mask.cpu().numpy().squeeze()
    active_mask = (graph_data.x.cpu() != 0).any(dim=1).numpy()

    return {
        'node_mask': node_mask,
        'edge_mask': edge_mask,
        'active_mask': active_mask,
        'edge_index': graph_data.edge_index.cpu().numpy(),
        'edge_attr': graph_data.edge_attr.cpu().numpy() if graph_data.edge_attr is not None else None,
        'prediction': explanation.prediction.cpu().numpy()
    }


def batch_explain_and_aggregate(model, explainer, data_dir, id2idx, idx2id, device,
                                  num_benign=50, num_injected=50, top_k=20):
    """
    Explain multiple graphs and aggregate node importance scores.
    Returns discriminative nodes that distinguish injected from benign.
    """
    print(f"\n{'='*80}")
    print(f"BATCH EXPLANATION: {num_benign} benign + {num_injected} injected graphs")
    print(f"{'='*80}\n")

    # Collect all explanations
    benign_scores = defaultdict(list)  # node_id -> list of scores
    injected_scores = defaultdict(list)

    benign_edge_scores = defaultdict(list)  # (src_id, dst_id) -> list of scores
    injected_edge_scores = defaultdict(list)

    # Process benign graphs
    benign_files = sorted(glob.glob(os.path.join(data_dir, "benign/*.json")))[:num_benign]
    print(f"[...] Explaining {len(benign_files)} benign graphs...")
    for i, graph_path in enumerate(benign_files):
        print(f"  [{i+1}/{len(benign_files)}] {os.path.basename(graph_path)}")
        try:
            graph_dict = build_plain_sample(graph_path, id2idx, label=0,
                                             node_feature_keys=NODE_FEATURE_KEYS,
                                             add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                             use_float16=False)
            graph_data = Data(**graph_dict).to(device)
            result = explain_single_graph(model, explainer, graph_data, idx2id, device)

            # Collect node scores (only for active nodes)
            for idx in range(len(result['node_mask'])):
                if result['active_mask'][idx]:
                    node_id = idx2id[idx]
                    benign_scores[node_id].append(float(result['node_mask'][idx]))

            # Collect edge scores
            for edge_idx in range(result['edge_index'].shape[1]):
                src_idx = result['edge_index'][0, edge_idx]
                dst_idx = result['edge_index'][1, edge_idx]
                src_id = idx2id[src_idx]
                dst_id = idx2id[dst_idx]
                edge_id = (src_id, dst_id)
                benign_edge_scores[edge_id].append(float(result['edge_mask'][edge_idx]))

        except Exception as e:
            print(f"    [!] Failed: {e}")
            continue

    # Process injected graphs
    injected_files = sorted(glob.glob(os.path.join(data_dir, "injected/*.json")))[:num_injected]
    print(f"\n[...] Explaining {len(injected_files)} injected graphs...")
    for i, graph_path in enumerate(injected_files):
        print(f"  [{i+1}/{len(injected_files)}] {os.path.basename(graph_path)}")
        try:
            graph_dict = build_plain_sample(graph_path, id2idx, label=1,
                                             node_feature_keys=NODE_FEATURE_KEYS,
                                             add_is_active_flag=ADD_IS_ACTIVE_FLAG,
                                             use_float16=False)
            graph_data = Data(**graph_dict).to(device)
            result = explain_single_graph(model, explainer, graph_data, idx2id, device)

            # Collect node scores
            for idx in range(len(result['node_mask'])):
                if result['active_mask'][idx]:
                    node_id = idx2id[idx]
                    injected_scores[node_id].append(float(result['node_mask'][idx]))

            # Collect edge scores
            for edge_idx in range(result['edge_index'].shape[1]):
                src_idx = result['edge_index'][0, edge_idx]
                dst_idx = result['edge_index'][1, edge_idx]
                src_id = idx2id[src_idx]
                dst_id = idx2id[dst_idx]
                edge_id = (src_id, dst_id)
                injected_edge_scores[edge_id].append(float(result['edge_mask'][edge_idx]))

        except Exception as e:
            print(f"    [!] Failed: {e}")
            continue

    # Aggregate scores: compute mean and discriminative score
    print(f"\n[...] Aggregating scores across all graphs...")

    all_node_ids = set(benign_scores.keys()) | set(injected_scores.keys())
    node_stats = []

    for node_id in all_node_ids:
        benign_vals = benign_scores.get(node_id, [])
        injected_vals = injected_scores.get(node_id, [])

        benign_mean = np.mean(benign_vals) if benign_vals else 0.0
        injected_mean = np.mean(injected_vals) if injected_vals else 0.0
        discriminative_score = injected_mean - benign_mean

        # Calculate standard deviations for robustness
        benign_std = np.std(benign_vals) if len(benign_vals) > 1 else 0.0
        injected_std = np.std(injected_vals) if len(injected_vals) > 1 else 0.0

        # Compute effect size (Cohen's d) - measures strength of difference
        pooled_std = np.sqrt((benign_std**2 + injected_std**2) / 2) if (benign_std > 0 or injected_std > 0) else 1.0
        effect_size = discriminative_score / pooled_std if pooled_std > 0 else 0.0

        # Frequency ratios
        freq_inj_ratio = len(injected_vals) / len(injected_files) if injected_files else 0.0
        freq_ben_ratio = len(benign_vals) / len(benign_files) if benign_files else 0.0

        # Combined score: balances frequency and importance
        # Favors nodes that appear frequently in injected, rarely in benign, with high importance
        combined_score = discriminative_score * freq_inj_ratio * (1.0 - freq_ben_ratio * 0.5)

        node_stats.append({
            'node_id': node_id,
            'injected_mean': injected_mean,
            'benign_mean': benign_mean,
            'discriminative_score': discriminative_score,
            'freq_injected': len(injected_vals),
            'freq_benign': len(benign_vals),
            'freq_inj_ratio': freq_inj_ratio,
            'freq_ben_ratio': freq_ben_ratio,
            'total_injected': len(injected_files),
            'total_benign': len(benign_files),
            'effect_size': effect_size,
            'combined_score': combined_score,
        })

    # Sort by combined score (default)
    node_stats.sort(key=lambda x: x['combined_score'], reverse=True)

    # Aggregate edge scores
    all_edge_ids = set(benign_edge_scores.keys()) | set(injected_edge_scores.keys())
    edge_stats = []

    for edge_id in all_edge_ids:
        benign_vals = benign_edge_scores.get(edge_id, [])
        injected_vals = injected_edge_scores.get(edge_id, [])

        benign_mean = np.mean(benign_vals) if benign_vals else 0.0
        injected_mean = np.mean(injected_vals) if injected_vals else 0.0
        discriminative_score = injected_mean - benign_mean

        edge_stats.append({
            'src_id': edge_id[0],
            'dst_id': edge_id[1],
            'injected_mean': injected_mean,
            'benign_mean': benign_mean,
            'discriminative_score': discriminative_score,
            'freq_injected': len(injected_vals),
            'freq_benign': len(benign_vals),
        })

    edge_stats.sort(key=lambda x: x['discriminative_score'], reverse=True)

    return node_stats, edge_stats


def main():
    ap = argparse.ArgumentParser(description="Explain GCN predictions using GNNExplainer")
    ap.add_argument("model_path", type=str, help="Path to trained model .pt file")
    ap.add_argument("data_dir", type=str, help="Path to data directory containing benign/ and injected/")
    ap.add_argument("--graph-file", type=str, default=None, help="Specific JSON file to explain (e.g., benign/0.json)")
    ap.add_argument("--vocabulary", type=str, default=None, help="Path to vocabulary.json")
    ap.add_argument("--explainer-epochs", type=int, default=100)
    ap.add_argument("--output-dir", type=str, default="explanations")
    ap.add_argument("--cuda", action="store_true")

    # Batch aggregation mode
    ap.add_argument("--batch", action="store_true", help="Run batch explanation and aggregation")
    ap.add_argument("--num-benign", type=int, default=50, help="Number of benign graphs to explain")
    ap.add_argument("--num-injected", type=int, default=50, help="Number of injected graphs to explain")
    ap.add_argument("--top-k", type=int, default=20, help="Number of top discriminative nodes/edges to report")
    ap.add_argument("--min-freq", type=float, default=0.1, help="Minimum frequency ratio (0.0-1.0) for a node to be considered (default: 0.1 = 10%%)")
    ap.add_argument("--rank-by", type=str, default="combined", choices=["combined", "discriminative", "frequency", "effect_size"],
                    help="Ranking strategy: combined (freq*score), discriminative (avg difference), frequency (appearance count), effect_size (Cohen's d)")

    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    in_dim = checkpoint["in_dim"]
    num_classes = checkpoint["num_classes"]
    hidden = checkpoint.get("hidden", 128)

    # Check if this is an old checkpoint (has conv1/conv2 instead of num_layers)
    if "num_layers" not in checkpoint:
        print("[!] Old checkpoint detected - using legacy 2-layer model")
        # Use old architecture
        class OldGCNGraphClassifier(torch.nn.Module):
            def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 2):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden)
                self.conv2 = GCNConv(hidden, hidden)
                self.lin = torch.nn.Linear(hidden, num_classes)

            def forward(self, x, edge_index, batch, edge_attr=None):
                if x.dtype == torch.float16:
                    x = x.float()
                edge_weight = None
                if edge_attr is not None:
                    edge_weight = edge_attr.abs().squeeze(-1)
                    if edge_weight.dtype == torch.float16:
                        edge_weight = edge_weight.float()

                x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
                x = F.dropout(x, p=0.5, training=self.training)
                x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
                x = global_mean_pool(x, batch)
                return F.log_softmax(self.lin(x), dim=-1)

        model = OldGCNGraphClassifier(in_dim=in_dim, hidden=hidden, num_classes=num_classes).to(device)
    else:
        # New architecture
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

    # Create explainer (shared for single and batch mode)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=args.explainer_epochs),
        explanation_type="model",
        node_mask_type="object",  # Per-node importance (not per-feature)
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="log_probs",
        ),
    )

    # BATCH MODE: Explain multiple graphs and aggregate
    if args.batch:
        node_stats, edge_stats = batch_explain_and_aggregate(
            model, explainer, args.data_dir, id2idx, idx2id, device,
            num_benign=args.num_benign,
            num_injected=args.num_injected,
            top_k=args.top_k
        )

        # Filter by minimum frequency
        print(f"\n[...] Filtering nodes with min frequency ratio: {args.min_freq:.2f}")
        filtered_nodes = [
            stat for stat in node_stats
            if stat['freq_inj_ratio'] >= args.min_freq
        ]
        print(f"[✓] Kept {len(filtered_nodes)}/{len(node_stats)} nodes after frequency filter")

        # Re-rank by chosen strategy
        rank_key_map = {
            'combined': 'combined_score',
            'discriminative': 'discriminative_score',
            'frequency': 'freq_inj_ratio',
            'effect_size': 'effect_size',
        }
        rank_key = rank_key_map[args.rank_by]
        filtered_nodes.sort(key=lambda x: x[rank_key], reverse=True)
        print(f"[✓] Ranked by: {args.rank_by}")

        # Print results
        print(f"\n{'='*120}")
        print(f"TOP {args.top_k} DISCRIMINATIVE NODES (ranked by {args.rank_by}, min_freq={args.min_freq:.0%}):")
        print(f"{'='*120}")
        print(f"{'Node ID':<40} | {'Inj Avg':>8} | {'Ben Avg':>8} | {'Diff':>8} | {'Effect':>8} | {'Freq Inj':>12} | {'Freq Ben':>12}")
        print(f"{'-'*120}")
        for stat in filtered_nodes[:args.top_k]:
            print(f"{stat['node_id']:<40} | {stat['injected_mean']:>8.4f} | {stat['benign_mean']:>8.4f} | "
                  f"{stat['discriminative_score']:>+8.4f} | {stat['effect_size']:>+8.2f} | "
                  f"{stat['freq_injected']:>4}/{stat['total_injected']:<5} | "
                  f"{stat['freq_benign']:>4}/{stat['total_benign']:<5}")

        print(f"\n{'='*100}")
        print(f"TOP {args.top_k} DISCRIMINATIVE EDGES (favor injected over benign):")
        print(f"{'='*100}")
        print(f"{'Source':<30} -> {'Destination':<30} | {'Inj Avg':>8} | {'Ben Avg':>8} | {'Diff':>8}")
        print(f"{'-'*100}")
        for stat in edge_stats[:args.top_k]:
            print(f"{stat['src_id']:<30} -> {stat['dst_id']:<30} | {stat['injected_mean']:>8.4f} | "
                  f"{stat['benign_mean']:>8.4f} | {stat['discriminative_score']:>+8.4f}")

        # Save aggregated results
        os.makedirs(args.output_dir, exist_ok=True)
        agg_file = os.path.join(args.output_dir, "discriminative_nodes_edges.txt")
        with open(agg_file, 'w') as f:
            f.write(f"BATCH EXPLANATION RESULTS\n")
            f.write(f"Benign graphs: {args.num_benign}, Injected graphs: {args.num_injected}\n")
            f.write(f"Explainer epochs: {args.explainer_epochs}\n")
            f.write(f"Ranking strategy: {args.rank_by}\n")
            f.write(f"Minimum frequency: {args.min_freq:.0%}\n")
            f.write(f"Nodes after filtering: {len(filtered_nodes)}/{len(node_stats)}\n\n")

            f.write(f"TOP {args.top_k} DISCRIMINATIVE NODES:\n")
            f.write(f"{'='*120}\n")
            f.write(f"{'Node ID':<40} | {'Inj Avg':>8} | {'Ben Avg':>8} | {'Diff':>8} | {'Effect':>8} | {'Freq Inj':>12} | {'Freq Ben':>12}\n")
            f.write(f"{'-'*120}\n")
            for stat in filtered_nodes[:args.top_k]:
                f.write(f"{stat['node_id']:<40} | {stat['injected_mean']:>8.4f} | {stat['benign_mean']:>8.4f} | "
                        f"{stat['discriminative_score']:>+8.4f} | {stat['effect_size']:>+8.2f} | "
                        f"{stat['freq_injected']:>4}/{stat['total_injected']:<5} | "
                        f"{stat['freq_benign']:>4}/{stat['total_benign']:<5}\n")

            f.write(f"\n\nTOP {args.top_k} DISCRIMINATIVE EDGES:\n")
            f.write(f"{'='*100}\n")
            f.write(f"{'Source':<30} -> {'Destination':<30} | {'Inj Avg':>8} | {'Ben Avg':>8} | {'Diff':>8}\n")
            f.write(f"{'-'*100}\n")
            for stat in edge_stats[:args.top_k]:
                f.write(f"{stat['src_id']:<30} -> {stat['dst_id']:<30} | {stat['injected_mean']:>8.4f} | "
                        f"{stat['benign_mean']:>8.4f} | {stat['discriminative_score']:>+8.4f}\n")

        print(f"\n[✓] Saved aggregated results to {agg_file}")
        return

    # SINGLE GRAPH MODE: Explain one specific graph
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

    # Generate explanation
    print(f"[...] Generating explanation (this may take a while)...")
    result = explain_single_graph(model, explainer, to_explain, idx2id, device)
    print(f"[✓] Generated explanation")

    # Extract important nodes and edges with their IDs
    node_mask = result['node_mask']
    edge_mask = result['edge_mask']
    active_mask = result['active_mask']
    edge_index_np = result['edge_index']
    num_active = active_mask.sum()

    print(f"\n[+] Graph has {num_active} active nodes out of {len(active_mask)} total")
    print(f"[+] Node mask shape: {node_mask.shape}, Edge mask shape: {edge_mask.shape}")

    # Get top-k important nodes (among ALL nodes, including zeros)
    top_k = min(50, len(node_mask))
    top_node_indices = np.argsort(node_mask)[::-1][:top_k]

    # Filter to only active nodes
    top_active_nodes = [(idx, float(node_mask[idx])) for idx in top_node_indices if active_mask[idx]][:20]

    print(f"\n{'='*80}")
    print(f"TOP 20 IMPORTANT NODES (with actual activity):")
    print(f"{'='*80}")
    for idx, score in top_active_nodes:
        node_id = idx2id[idx]
        features = to_explain.x[idx].cpu().numpy()
        print(f"  {node_id:40s} | score: {score:.4f} | features: {features}")

    # Get top-k important edges
    top_k_edges = min(20, len(edge_mask))
    top_edge_indices = np.argsort(edge_mask)[::-1][:top_k_edges]

    print(f"\n{'='*80}")
    print(f"TOP 20 IMPORTANT EDGES:")
    print(f"{'='*80}")
    for edge_idx in top_edge_indices:
        src_idx = edge_index_np[0, edge_idx]
        dst_idx = edge_index_np[1, edge_idx]
        src_id = idx2id[src_idx]
        dst_id = idx2id[dst_idx]
        score = edge_mask[edge_idx]
        edge_weight = result['edge_attr'][edge_idx] if result['edge_attr'] is not None else 1.0
        print(f"  {src_id:30s} -> {dst_id:30s} | score: {score:.4f} | weight: {edge_weight:.4f}")

    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)

    results_file = os.path.join(args.output_dir, "important_nodes_edges.txt")
    with open(results_file, 'w') as f:
        f.write(f"Explanation for: {graph_path}\n")
        f.write(f"Model prediction: {result['prediction']}\n\n")

        f.write("TOP 20 IMPORTANT NODES:\n")
        f.write("="*80 + "\n")
        for idx, score in top_active_nodes:
            node_id = idx2id[idx]
            features = to_explain.x[idx].cpu().numpy()
            f.write(f"{node_id:40s} | score: {score:.4f} | features: {features}\n")

        f.write("\n\nTOP 20 IMPORTANT EDGES:\n")
        f.write("="*80 + "\n")
        for edge_idx in top_edge_indices:
            src_idx = edge_index_np[0, edge_idx]
            dst_idx = edge_index_np[1, edge_idx]
            src_id = idx2id[src_idx]
            dst_id = idx2id[dst_idx]
            score = edge_mask[edge_idx]
            edge_weight = result['edge_attr'][edge_idx] if result['edge_attr'] is not None else 1.0
            f.write(f"{src_id:30s} -> {dst_id:30s} | score: {score:.4f} | weight: {edge_weight:.4f}\n")

    print(f"\n[✓] Saved important nodes/edges to {results_file}")

if __name__ == "__main__":
    main()
