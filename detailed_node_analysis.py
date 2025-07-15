#!/usr/bin/env python3
"""
Detailed Analysis of GNN Explainer Results with Node-Level Insights
This script analyzes the explainer output to understand specific nodes and patterns
the model learned to distinguish injected from benign graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
import sys
sys.path.append('.')

# Import our semantic converter
from convert_attribution_graphs_semantic import SemanticAttributionGraphConverter

# Import Neuronpedia integration
try:
    from neuronpedia_integration import NeuronpediaClient, enrich_analysis_with_neuronpedia, create_interpretation_report
    NEURONPEDIA_AVAILABLE = True
except ImportError:
    NEURONPEDIA_AVAILABLE = False
    print("Neuronpedia integration not available. Install requests to enable feature interpretations.")

class DetailedNodeAnalyzer:
    def __init__(self, log_dir: str = "log", data_dir: str = "data_small"):
        self.log_dir = Path(log_dir)
        self.data_dir = Path(data_dir)
        self.converter = SemanticAttributionGraphConverter()
        
        # Feature interpretation
        self.feature_names = [
            'influence', 'activation', 'layer', 'ctx_idx', 'feature',
            'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit'
        ]
        
        # Node type mappings
        self.node_type_map = {
            'cross layer transcoder': 'Transcoder Feature',
            'mlp reconstruction error': 'MLP Error',
            'embedding': 'Token Embedding',
            'logit': 'Output Logit',
            'unknown': 'Unknown'
        }
        
    def load_and_analyze_original_graphs(self) -> Tuple[List[Dict], List[Dict]]:
        """Load original JSON graphs to understand node semantics"""
        print("=== LOADING ORIGINAL GRAPHS ===")
        
        benign_graphs = []
        injected_graphs = []
        
        # Load benign graphs
        benign_dir = self.data_dir / "benign"
        if benign_dir.exists():
            for json_file in sorted(benign_dir.glob("*.json")):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    benign_graphs.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        # Load injected graphs  
        injected_dir = self.data_dir / "injected"
        if injected_dir.exists():
            for json_file in sorted(injected_dir.glob("*.json")):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    injected_graphs.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(benign_graphs)} benign and {len(injected_graphs)} injected graphs")
        return benign_graphs, injected_graphs
    
    def analyze_node_types(self, benign_graphs: List[Dict], injected_graphs: List[Dict]) -> Dict:
        """Analyze node type distributions between benign and injected graphs"""
        print("\\n=== NODE TYPE ANALYSIS ===")
        
        def count_node_types(graphs, label):
            type_counts = Counter()
            layer_counts = defaultdict(Counter)
            feature_counts = defaultdict(Counter)
            
            for graph in graphs:
                nodes = graph.get("nodes", [])
                for node in nodes:
                    feature_type = node.get("feature_type", "unknown")
                    layer = node.get("layer", "unknown")
                    feature_id = node.get("feature", "unknown")
                    
                    type_counts[feature_type] += 1
                    layer_counts[feature_type][layer] += 1
                    if feature_type == "cross layer transcoder":
                        feature_counts[layer][feature_id] += 1
            
            return type_counts, layer_counts, feature_counts
        
        benign_types, benign_layers, benign_features = count_node_types(benign_graphs, "benign")
        injected_types, injected_layers, injected_features = count_node_types(injected_graphs, "injected")
        
        print("Node type distributions:")
        print(f"{'Type':<25} {'Benign':<10} {'Injected':<10} {'Difference':<10}")
        print("-" * 60)
        
        all_types = set(benign_types.keys()) | set(injected_types.keys())
        for node_type in sorted(all_types):
            b_count = benign_types.get(node_type, 0)
            i_count = injected_types.get(node_type, 0)
            diff = i_count - b_count
            print(f"{node_type:<25} {b_count:<10} {i_count:<10} {diff:<10}")
        
        return {
            'benign_types': benign_types,
            'injected_types': injected_types,
            'benign_layers': benign_layers,
            'injected_layers': injected_layers,
            'benign_features': benign_features,
            'injected_features': injected_features
        }
    
    def load_masked_adjacency_with_mapping(self) -> Tuple[np.ndarray, List[Dict]]:
        """Load masked adjacency matrix and create node mapping"""
        print("\\n=== LOADING MASKED ADJACENCY WITH NODE MAPPING ===")
        
        # Find the masked adjacency file
        masked_files = list(self.log_dir.glob("masked_adj_*.npy"))
        if not masked_files:
            raise FileNotFoundError("No masked adjacency files found")
        
        adj_matrix = np.load(masked_files[0])
        print(f"Loaded adjacency matrix shape: {adj_matrix.shape}")
        
        # Try to find cached graphs to get node mappings
        possible_cache_dirs = [
            "/home/sk2959/palmer_scratch/gnn_explainer_cache_semantic",
            "../gnn_explainer_cache_semantic", 
            "gnn_explainer_cache_semantic",
        ]
        
        node_mapping = []
        for cache_dir in possible_cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                cache_files = list(cache_path.glob("*.pkl"))
                if cache_files:
                    with open(cache_files[0], 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Get node mappings from the first graph
                    if cached_data['graphs']:
                        first_graph = cached_data['graphs'][0]
                        for node_id in first_graph.nodes():
                            node_mapping.append({
                                'graph_node_id': node_id,
                                'features': first_graph.nodes[node_id]['feat']
                            })
                    break
        
        print(f"Found {len(node_mapping)} node mappings")
        return adj_matrix, node_mapping
    
    def analyze_important_nodes(self, adj_matrix: np.ndarray, node_mapping: List[Dict]) -> pd.DataFrame:
        """Analyze which specific nodes are most important in the explanation"""
        print("\\n=== IMPORTANT NODE ANALYSIS ===")
        
        # Skip the complex node mapping - use processed feature information instead
        print("Using processed feature information directly (more reliable than mapping back to original JSON)")
        
        # Calculate node importance metrics
        node_importance = []
        
        for i in range(adj_matrix.shape[0]):
            # Total influence (sum of absolute weights)
            in_influence = np.sum(np.abs(adj_matrix[:, i]))
            out_influence = np.sum(np.abs(adj_matrix[i, :]))
            total_influence = in_influence + out_influence
            
            # Positive vs negative influence
            pos_in = np.sum(adj_matrix[adj_matrix[:, i] > 0, i])
            neg_in = np.sum(adj_matrix[adj_matrix[:, i] < 0, i])
            pos_out = np.sum(adj_matrix[i, adj_matrix[i, :] > 0])
            neg_out = np.sum(adj_matrix[i, adj_matrix[i, :] < 0])
            
            # Connection counts
            in_connections = np.count_nonzero(adj_matrix[:, i])
            out_connections = np.count_nonzero(adj_matrix[i, :])
            
            node_info = {
                'node_idx': i,
                'total_influence': total_influence,
                'in_influence': in_influence,
                'out_influence': out_influence,
                'pos_in_influence': pos_in,
                'neg_in_influence': neg_in,
                'pos_out_influence': pos_out,
                'neg_out_influence': neg_out,
                'in_connections': in_connections,
                'out_connections': out_connections,
                'total_connections': in_connections + out_connections
            }
            
            # Initialize with defaults
            node_info.update({
                'feature_id': 'unknown',
                'node_id': f'node_{i}',
                'original_layer': 'unknown',
                'original_ctx_idx': 'unknown',
                'node_type': 'unknown'
            })
            
            # Add processed feature information if available
            if i < len(node_mapping):
                features = node_mapping[i]['features']
                node_info.update({
                    'influence_feat': features[0],
                    'activation_feat': features[1], 
                    'layer_feat': int(features[2]),
                    'ctx_idx_feat': int(features[3]),
                    'processed_feature_id': features[4],
                    'is_transcoder': bool(features[5]),
                    'is_mlp_error': bool(features[6]),
                    'is_embedding': bool(features[7]),
                    'is_target_logit': bool(features[8])
                })
                
                # Use processed feature information to set feature_id
                # This will be reverse-normalized later in neuronpedia_integration.py
                node_info['feature_id'] = features[4]  # Use processed feature ID
                
                # Determine node type from processed features
                if features[5] > 0.5:
                    node_info['node_type'] = 'transcoder'
                elif features[6] > 0.5:
                    node_info['node_type'] = 'mlp_error'
                elif features[7] > 0.5:
                    node_info['node_type'] = 'embedding'
                elif features[8] > 0.5:
                    node_info['node_type'] = 'target_logit'
                else:
                    node_info['node_type'] = 'unknown'
            
            node_importance.append(node_info)
        
        df = pd.DataFrame(node_importance)
        df = df.sort_values('total_influence', ascending=False)
        
        print(f"Created DataFrame with {len(df)} nodes and columns: {list(df.columns)}")
        print("Top 20 most important nodes:")
        # Show original feature IDs and node information
        display_cols = ['node_idx', 'node_type', 'total_influence', 'original_layer', 'feature_id', 'node_id', 'total_connections']
        available_cols = [col for col in display_cols if col in df.columns]
        
        if available_cols:
            print(df[available_cols].head(20))
        else:
            print(df[['node_idx', 'total_influence', 'total_connections']].head(20))
        
        return df
    
    def analyze_node_type_importance(self, df: pd.DataFrame) -> Dict:
        """Analyze importance by node type"""
        print("\\n=== NODE TYPE IMPORTANCE ANALYSIS ===")
        
        if 'node_type' not in df.columns:
            print("Node type information not available")
            return {}
        
        # Group by node type
        type_analysis = df.groupby('node_type').agg({
            'total_influence': ['count', 'mean', 'sum', 'std'],
            'total_connections': ['mean', 'sum'],
            'layer_feat': 'mean',
            'pos_out_influence': 'mean',
            'neg_out_influence': 'mean'
        }).round(4)
        
        print("Importance by node type:")
        print(type_analysis)
        
        # Analyze layer distribution
        print("\\nLayer distribution by node type:")
        layer_dist = df.groupby(['node_type', 'layer_feat']).size().unstack(fill_value=0)
        print(layer_dist)
        
        return {
            'type_analysis': type_analysis,
            'layer_distribution': layer_dist
        }
    
    def analyze_critical_pathways(self, adj_matrix: np.ndarray, node_df: pd.DataFrame) -> Dict:
        """Analyze critical pathways in the attribution graph"""
        print("\\n=== CRITICAL PATHWAY ANALYSIS ===")
        
        # Find strongest connections
        strong_edges = []
        threshold = np.percentile(np.abs(adj_matrix[adj_matrix != 0]), 90)  # Top 10% of edges
        
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if abs(adj_matrix[i, j]) > threshold:
                    source_type = node_df.iloc[j].get('node_type', 'unknown') if j < len(node_df) else 'unknown'
                    target_type = node_df.iloc[i].get('node_type', 'unknown') if i < len(node_df) else 'unknown'
                    
                    strong_edges.append({
                        'source': j,
                        'target': i,
                        'weight': adj_matrix[i, j],
                        'abs_weight': abs(adj_matrix[i, j]),
                        'source_type': source_type,
                        'target_type': target_type,
                        'edge_type': f"{source_type} -> {target_type}"
                    })
        
        edge_df = pd.DataFrame(strong_edges)
        
        print(f"Found {len(strong_edges)} strong connections (>{threshold:.4f})")
        
        if not edge_df.empty:
            print("\\nTop edge types by count:")
            edge_type_counts = edge_df['edge_type'].value_counts()
            print(edge_type_counts.head(10))
            
            print("\\nStrongest individual edges:")
            top_edges = edge_df.nlargest(10, 'abs_weight')
            print(top_edges[['source', 'target', 'weight', 'edge_type']])
        
        return {
            'strong_edges': strong_edges,
            'edge_types': edge_type_counts.to_dict() if not edge_df.empty else {},
            'threshold': threshold
        }
    
    def create_detailed_visualizations(self, node_df: pd.DataFrame, pathway_analysis: Dict):
        """Create detailed visualizations of the analysis"""
        print("\\n=== CREATING DETAILED VISUALIZATIONS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Node type distribution
        if 'node_type' in node_df.columns:
            type_counts = node_df['node_type'].value_counts()
            axes[0,0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[0,0].set_title('Node Type Distribution')
        else:
            axes[0,0].text(0.5, 0.5, 'Node type\\ninformation\\nnot available', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Node Type Distribution')
        
        # 2. Influence by node type
        if 'node_type' in node_df.columns:
            sns.boxplot(data=node_df, x='node_type', y='total_influence', ax=axes[0,1])
            axes[0,1].set_title('Influence Distribution by Node Type')
            axes[0,1].tick_params(axis='x', rotation=45)
        else:
            axes[0,1].hist(node_df['total_influence'], bins=50, alpha=0.7)
            axes[0,1].set_title('Overall Influence Distribution')
        
        # 3. Layer distribution
        if 'layer_feat' in node_df.columns:
            layer_counts = node_df['layer_feat'].value_counts().sort_index()
            axes[0,2].bar(layer_counts.index, layer_counts.values)
            axes[0,2].set_title('Important Nodes by Layer')
            axes[0,2].set_xlabel('Layer')
            axes[0,2].set_ylabel('Count')
        else:
            axes[0,2].text(0.5, 0.5, 'Layer\\ninformation\\nnot available', 
                          ha='center', va='center', transform=axes[0,2].transAxes)
        
        # 4. Edge type distribution
        if pathway_analysis['edge_types']:
            edge_types = list(pathway_analysis['edge_types'].keys())[:10]
            edge_counts = [pathway_analysis['edge_types'][et] for et in edge_types]
            axes[1,0].barh(edge_types, edge_counts)
            axes[1,0].set_title('Top Edge Types in Critical Pathways')
            axes[1,0].set_xlabel('Count')
        else:
            axes[1,0].text(0.5, 0.5, 'Edge type\\ninformation\\nnot available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 5. Positive vs negative influence
        if 'pos_out_influence' in node_df.columns and 'neg_out_influence' in node_df.columns:
            axes[1,1].scatter(node_df['pos_out_influence'], node_df['neg_out_influence'], 
                             alpha=0.6, s=20)
            axes[1,1].set_xlabel('Positive Outgoing Influence')
            axes[1,1].set_ylabel('Negative Outgoing Influence')
            axes[1,1].set_title('Positive vs Negative Influence')
            axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        else:
            axes[1,1].text(0.5, 0.5, 'Influence\\nbreakdown\\nnot available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        # 6. Top nodes by influence
        top_nodes = node_df.head(20)
        axes[1,2].barh(range(len(top_nodes)), top_nodes['total_influence'])
        axes[1,2].set_title('Top 20 Nodes by Influence')
        axes[1,2].set_xlabel('Total Influence')
        axes[1,2].set_yticks(range(len(top_nodes)))
        axes[1,2].set_yticklabels([f"Node {idx}" for idx in top_nodes['node_idx']])
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'detailed_node_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {self.log_dir / 'detailed_node_analysis.png'}")
        
        return fig
    
    def run_full_analysis(self):
        """Run the complete detailed analysis"""
        print("="*80)
        print("DETAILED NODE-LEVEL EXPLAINER ANALYSIS")
        print("="*80)
        
        # 1. Load original graphs
        benign_graphs, injected_graphs = self.load_and_analyze_original_graphs()
        
        # 2. Analyze node types
        node_type_analysis = self.analyze_node_types(benign_graphs, injected_graphs)
        
        # 3. Load masked adjacency with node mapping
        adj_matrix, node_mapping = self.load_masked_adjacency_with_mapping()
        
        # 4. Analyze important nodes
        node_df = self.analyze_important_nodes(adj_matrix, node_mapping)
        
        # 5. Analyze node type importance
        type_importance = self.analyze_node_type_importance(node_df)
        
        # 6. Analyze critical pathways
        pathway_analysis = self.analyze_critical_pathways(adj_matrix, node_df)
        
        # 7. Enrich with Neuronpedia interpretations (if available)
        if NEURONPEDIA_AVAILABLE:
            print("\\n=== ENRICHING WITH NEURONPEDIA INTERPRETATIONS ===")
            try:
                client = NeuronpediaClient()
                node_df = enrich_analysis_with_neuronpedia(node_df, client)
                
                # Create interpretation report
                create_interpretation_report(node_df, self.log_dir / "feature_interpretations.html")
                print(f"Feature interpretation report saved to: {self.log_dir / 'feature_interpretations.html'}")
                
            except Exception as e:
                print(f"Neuronpedia enrichment failed: {e}")
                print("Continuing without interpretations...")
        else:
            print("\\nNeuronpedia integration not available. Skipping feature interpretations.")
        
        # 8. Create visualizations
        fig = self.create_detailed_visualizations(node_df, pathway_analysis)
        
        # 9. Save detailed results
        results = {
            'node_type_analysis': {k: dict(v) if hasattr(v, 'items') else v 
                                 for k, v in node_type_analysis.items()},
            'pathway_analysis': pathway_analysis,
            'top_nodes': node_df.head(50).to_dict('records'),
            'adjacency_stats': {
                'shape': adj_matrix.shape,
                'nonzero_edges': int(np.count_nonzero(adj_matrix)),
                'sparsity': float(np.count_nonzero(adj_matrix) / adj_matrix.size)
            }
        }
        
        results_file = self.log_dir / 'detailed_node_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\nDetailed analysis complete! Results saved to: {results_file}")
        
        return results

def main():
    analyzer = DetailedNodeAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\\n" + "="*80)
    print("SUMMARY OF KEY FINDINGS")
    print("="*80)
    
    if results['adjacency_stats']:
        stats = results['adjacency_stats']
        print(f"Graph size: {stats['shape'][0]} nodes")
        print(f"Important connections: {stats['nonzero_edges']} edges")
        print(f"Sparsity: {stats['sparsity']:.6f}")
    
    if results['top_nodes']:
        print(f"\\nTop 5 most important nodes:")
        for i, node in enumerate(results['top_nodes'][:5]):
            node_type = node.get('node_type', 'unknown')
            layer = node.get('original_layer', node.get('layer_feat', 'unknown'))
            feature_id = node.get('feature_id', 'unknown')
            influence = node.get('total_influence', 0)
            
            # Add interpretation if available
            interpretation = ""
            if node.get('neuronpedia_description'):
                desc = node['neuronpedia_description'][:50] + "..." if len(node['neuronpedia_description']) > 50 else node['neuronpedia_description']
                interpretation = f" - {desc}"
            
            print(f"  {i+1}. Node {node['node_idx']}: {node_type} (Layer {layer}, Feature {feature_id}, Influence: {influence:.4f}){interpretation}")
    
    if results['pathway_analysis']['edge_types']:
        print(f"\\nTop 3 critical pathway types:")
        for i, (edge_type, count) in enumerate(list(results['pathway_analysis']['edge_types'].items())[:3]):
            print(f"  {i+1}. {edge_type}: {count} connections")
    
    print("\\nThis analysis reveals the specific attribution graph components")
    print("that the GNN model uses to detect prompt injection attacks.")

if __name__ == "__main__":
    main()