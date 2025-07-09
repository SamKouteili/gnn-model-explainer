#!/usr/bin/env python3
"""
Test the semantic attribution graph conversion
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from convert_attribution_graphs_semantic import create_gnn_explainer_splits_semantic

def test_semantic_conversion():
    """Test the semantic conversion process"""
    
    print("Testing semantic attribution graph conversion...")
    
    # Test data directory
    data_dir = "/Users/samkouteili/rose/circuits/data_small"
    
    # Test loading graphs with semantic processing
    print(f"Loading from: {data_dir}")
    train_graphs, val_graphs, test_graphs = create_gnn_explainer_splits_semantic(
        data_dir, 
        max_files=5,  # Small test
        test_size=0.4,
        val_size=0.2
    )
    
    print(f"Split results:")
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val: {len(val_graphs)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    
    if len(train_graphs) == 0:
        print("No graphs loaded - check data directory path")
        return False
    
    # Test a single graph
    G = train_graphs[0]
    print(f"\nSample semantic graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Label: {G.graph['label']}")
    print(f"  Feature dim: {G.graph.get('feat_dim', 'not set')}")
    
    if G.number_of_nodes() > 0:
        feat = G.nodes[0]['feat']
        print(f"  Sample node features shape: {feat.shape}")
        print(f"  Sample node features: {feat}")
        
        # Check feature interpretation
        print(f"  Feature interpretation:")
        print(f"    Influence: {feat[0]:.6f}")
        print(f"    Activation: {feat[1]:.6f}")
        print(f"    Layer: {feat[2]:.0f}")
        print(f"    Context index: {feat[3]:.0f}")
        print(f"    Feature value: {feat[4]:.6f}")
        print(f"    Is cross-layer transcoder: {feat[5]:.0f}")
        print(f"    Is MLP error: {feat[6]:.0f}")
        print(f"    Is embedding: {feat[7]:.0f}")
        print(f"    Is target logit: {feat[8]:.0f}")
    
    # Test edge weights
    if G.number_of_edges() > 0:
        edge_weights = [G[u][v]['weight'] for u, v in list(G.edges())[:10]]
        print(f"  Sample edge weights: {edge_weights[:5]}...")
        
        # Check edge weight statistics
        all_weights = [G[u][v]['weight'] for u, v in G.edges()]
        print(f"  Edge weight statistics:")
        print(f"    Count: {len(all_weights)}")
        print(f"    Min: {min(all_weights):.6f}")
        print(f"    Max: {max(all_weights):.6f}")
        print(f"    Negative count: {sum(1 for w in all_weights if w < 0)}")
        print(f"    Negative percentage: {sum(1 for w in all_weights if w < 0) / len(all_weights) * 100:.1f}%")
    
    # Check label distribution
    train_labels = [G.graph['label'] for G in train_graphs]
    val_labels = [G.graph['label'] for G in val_graphs]
    test_labels = [G.graph['label'] for G in test_graphs]
    
    print(f"\nLabel distribution:")
    print(f"  Train - Benign: {train_labels.count(0)}, Injected: {train_labels.count(1)}")
    print(f"  Val - Benign: {val_labels.count(0)}, Injected: {val_labels.count(1)}")
    print(f"  Test - Benign: {test_labels.count(0)}, Injected: {test_labels.count(1)}")
    
    # Compare with original conversion
    print(f"\nSemantic processing impact:")
    print(f"  Original node count (raw): 1693 (from JSON)")
    print(f"  Semantic node count (filtered): {G.number_of_nodes()}")
    print(f"  Filtered out: {1693 - G.number_of_nodes()} nodes (~{(1693 - G.number_of_nodes()) / 1693 * 100:.1f}%)")
    
    print(f"\n✅ Semantic conversion test passed!")
    return True

if __name__ == "__main__":
    test_semantic_conversion()