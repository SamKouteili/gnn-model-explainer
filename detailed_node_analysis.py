#!/usr/bin/env python3
"""
Detailed Analysis of GNN Explainer Results with Neuronpedia Integration

This script extracts the most important nodes from GNN explainer outputs,
enriches them with semantic information from the source graphs, and 
retrieves interpretations from Neuronpedia.
"""

import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from neuronpedia_integration import NeuronpediaClient


class DetailedNodeAnalyzer:
    def __init__(self, log_dir: str = "log", data_dir: str = "data_small"):
        self.log_dir = Path(log_dir)
        self.data_dir = Path(data_dir)
        self.client = NeuronpediaClient()
        
    def load_explainer_results(self) -> np.ndarray:
        """Load masked adjacency matrix from explainer output"""
        print("=== Loading Explainer Results ===")
        
        # Load masked adjacency matrix
        masked_files = list(self.log_dir.glob("masked_adj_*.npy"))
        if not masked_files:
            raise FileNotFoundError(f"No masked adjacency files found in {self.log_dir}")
        
        adj_matrix = np.load(masked_files[0])
        print(f"Loaded adjacency matrix: {adj_matrix.shape}")
        
        return adj_matrix
    
    def calculate_node_importance(self, adj_matrix: np.ndarray) -> pd.DataFrame:
        """Calculate importance metrics for each node"""
        print("=== Calculating Node Importance ===")
        
        node_data = []
        
        for i in range(adj_matrix.shape[0]):
            # Calculate influence metrics
            in_influence = np.sum(np.abs(adj_matrix[:, i]))
            out_influence = np.sum(np.abs(adj_matrix[i, :]))
            total_influence = in_influence + out_influence
            
            # Connection counts
            in_connections = np.count_nonzero(adj_matrix[:, i])
            out_connections = np.count_nonzero(adj_matrix[i, :])
            total_connections = in_connections + out_connections
            
            node_info = {
                'node_idx': i,
                'total_influence': total_influence,
                'in_influence': in_influence,
                'out_influence': out_influence,
                'total_connections': total_connections,
                'in_connections': in_connections,
                'out_connections': out_connections
            }
            
            node_data.append(node_info)
        
        df = pd.DataFrame(node_data)
        df = df.sort_values('total_influence', ascending=False)
        
        print(f"Calculated importance for {len(df)} nodes")
        return df
    
    def load_node_vocabulary(self, vocab_file: str) -> Tuple[Dict, Dict]:
        """Load pre-created node vocabulary from pickle file"""
        print("=== Loading Node Vocabulary ===")
        
        vocab_path = Path(vocab_file)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        
        print(f"Loading vocabulary from: {vocab_file}")
        with open(vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        
        node_vocab = vocab_data['node_vocab']
        node_index_map = vocab_data['node_index_map']
        stats = vocab_data.get('stats', {})
        
        print(f"✅ Vocabulary loaded successfully!")
        print(f"  Total nodes: {stats.get('total_nodes', len(node_index_map)):,}")
        print(f"  Unique (layer, feature_id) combinations: {len(node_vocab):,}")
        
        return node_vocab, node_index_map
    
    def is_valid_for_neuronpedia(self, node: Dict) -> bool:
        """Check if a transcoder node is valid for Neuronpedia lookup"""
        layer = node.get('layer')
        feature_id = node.get('feature_id')
        
        # Must have valid layer and feature_id
        if layer is None or feature_id is None:
            return False
            
        # Skip invalid layers
        if layer in [0, "0", "E"] or not str(layer).isdigit():
            return False
            
        # Feature ID must be in valid range for 16k SAEs
        if feature_id <= 0 or feature_id > 16384:
            return False
            
        return True
    
    def get_neuronpedia_interpretation(self, layer: int, feature_id: int) -> Optional[Dict]:
        """Get interpretation from Neuronpedia for a transcoder feature"""
        
        # All nodes are cross-layer transcoder, so always use transcoder SAE type
        # But also try other types as fallback
        sae_types_to_try = ['transcoder', 'res', 'att']
        
        for sae_type in sae_types_to_try:
            try:
                interpretation = self.client.get_feature_interpretation(
                    layer, int(feature_id), sae_type)
                
                if (interpretation and 
                    (interpretation.description != "No description available" or 
                     interpretation.explanation)):
                    
                    return {
                        'description': interpretation.description,
                        'explanation': interpretation.explanation,
                        'score': interpretation.auto_interp_score,
                        'sae_type': sae_type,
                        'top_tokens': [
                            f"{logit['token']} ({logit['logit_diff']:.3f})" 
                            for logit in (interpretation.top_logits or [])[:3]
                        ]
                    }
            except Exception:
                continue
                
        return None
    
    def enrich_with_semantic_info(self, df: pd.DataFrame, node_vocab: Dict, 
                                node_index_map: Dict) -> pd.DataFrame:
        """Enrich nodes with semantic information from source graphs"""
        print("=== Enriching Nodes with Semantic Information ===")
        
        # Add semantic information columns
        df['layer'] = None
        df['feature_id'] = None
        df['node_type'] = None
        df['activation'] = None
        df['ctx_idx'] = None
        df['source_file'] = None
        
        for idx, row in df.iterrows():
            node_idx = row['node_idx']
            
            # Try direct index mapping first (most reliable)
            if node_idx in node_index_map:
                node_info = node_index_map[node_idx]
                df.at[idx, 'layer'] = node_info['layer']
                df.at[idx, 'feature_id'] = node_info['feature_id']
                df.at[idx, 'node_type'] = node_info['node_type']
                df.at[idx, 'activation'] = node_info['activation']
                df.at[idx, 'ctx_idx'] = node_info['ctx_idx']
                df.at[idx, 'source_file'] = node_info['source_file']
        
        enriched_count = df['layer'].notna().sum()
        print(f"Successfully enriched {enriched_count}/{len(df)} nodes with semantic info")
        return df
    
    def enrich_with_neuronpedia(self, df: pd.DataFrame, 
                              max_nodes: int = 20) -> pd.DataFrame:
        """Enrich top nodes with Neuronpedia interpretations"""
        print(f"=== Enriching Top {max_nodes} Nodes with Neuronpedia ===")
        
        # Initialize interpretation columns
        df['neuronpedia_description'] = None
        df['neuronpedia_explanation'] = None
        df['neuronpedia_score'] = None
        df['neuronpedia_sae_type'] = None
        df['neuronpedia_top_tokens'] = None
        
        enriched_count = 0
        
        for idx, row in df.head(max_nodes).iterrows():
            if not self.is_valid_for_neuronpedia(row):
                print(f"  Skipping node {row['node_idx']}: Invalid for Neuronpedia lookup")
                continue
                
            print(f"  Checking node {row['node_idx']}: Layer {row['layer']}, "
                  f"Feature {row['feature_id']} (transcoder)")
            
            interpretation = self.get_neuronpedia_interpretation(
                int(row['layer']), row['feature_id'])
            
            if interpretation:
                df.at[idx, 'neuronpedia_description'] = interpretation['description']
                df.at[idx, 'neuronpedia_explanation'] = interpretation['explanation']
                df.at[idx, 'neuronpedia_score'] = interpretation['score']
                df.at[idx, 'neuronpedia_sae_type'] = interpretation['sae_type']
                df.at[idx, 'neuronpedia_top_tokens'] = '; '.join(interpretation['top_tokens'])
                
                enriched_count += 1
                print(f"    ✅ Found: {interpretation['description'][:60]}...")
            else:
                print(f"    ❌ No interpretation found")
        
        print(f"Successfully enriched {enriched_count}/{max_nodes} nodes")
        return df
    
    def create_summary_report(self, df: pd.DataFrame) -> str:
        """Create a summary report of the analysis"""
        report = []
        report.append("=" * 80)
        report.append("GNN EXPLAINER NODE ANALYSIS SUMMARY")
        report.append("=" * 80)
        
        # Overall statistics
        total_nodes = len(df)
        interpreted_nodes = df['neuronpedia_description'].notna().sum()
        
        report.append(f"Total nodes analyzed: {total_nodes}")
        report.append(f"Nodes with interpretations: {interpreted_nodes}")
        report.append("")
        
        # All nodes are transcoder type now
        report.append("Node type distribution:")
        report.append(f"  cross layer transcoder: {total_nodes} (100.0%)")
        report.append("")
        
        # Top interpreted nodes
        interpreted_df = df[df['neuronpedia_description'].notna()]
        if not interpreted_df.empty:
            report.append("TOP INTERPRETED NODES:")
            report.append("-" * 40)
            
            for i, (_, node) in enumerate(interpreted_df.head(10).iterrows(), 1):
                desc = node['neuronpedia_description']
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                    
                report.append(f"{i:2d}. Node {node['node_idx']} "
                            f"(Layer {node['layer']}, Feature {node['feature_id']})")
                report.append(f"    Type: cross layer transcoder")
                report.append(f"    Influence: {node['total_influence']:.4f}")
                report.append(f"    Description: {desc}")
                if node['neuronpedia_top_tokens']:
                    report.append(f"    Top tokens: {node['neuronpedia_top_tokens']}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, df: pd.DataFrame, report: str):
        """Save analysis results to files"""
        # Save detailed DataFrame
        csv_path = self.log_dir / "detailed_node_analysis_with_interpretations.csv"
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        # Save summary report
        report_path = self.log_dir / "node_analysis_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Summary report saved to: {report_path}")
        
        # Save top interpreted nodes as JSON for further analysis
        interpreted_nodes = df[df['neuronpedia_description'].notna()]
        if not interpreted_nodes.empty:
            json_path = self.log_dir / "interpreted_nodes.json"
            interpreted_data = interpreted_nodes.head(20).to_dict('records')
            
            with open(json_path, 'w') as f:
                json.dump(interpreted_data, f, indent=2, default=str)
            print(f"Interpreted nodes saved to: {json_path}")
    
    def run_analysis(self, vocab_file: str, max_nodes: int = 20) -> pd.DataFrame:
        """Run the complete analysis pipeline"""
        print("Starting GNN Explainer Node Analysis with Neuronpedia Integration")
        print("=" * 80)
        
        # Load explainer results
        adj_matrix = self.load_explainer_results()
        
        # Calculate node importance from adjacency matrix
        df = self.calculate_node_importance(adj_matrix)
        
        # Load pre-created node vocabulary
        node_vocab, node_index_map = self.load_node_vocabulary(vocab_file)
        
        # Enrich with semantic information
        df = self.enrich_with_semantic_info(df, node_vocab, node_index_map)
        
        # Show top nodes before Neuronpedia enrichment
        print(f"\nTop 10 transcoder nodes by influence:")
        display_cols = ['node_idx', 'layer', 'feature_id', 
                       'total_influence', 'total_connections']
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        # Enrich with Neuronpedia
        df = self.enrich_with_neuronpedia(df, max_nodes)
        
        # Create and display summary
        report = self.create_summary_report(df)
        print("\n" + report)
        
        # Save results
        self.save_results(df, report)
        
        return df


def main():
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python detailed_node_analysis.py <vocab_file> [max_nodes]")
        print("Example: python detailed_node_analysis.py node_vocabulary.pkl 20")
        print()
        print("First create the vocabulary with:")
        print("  python create_node_vocabulary.py /path/to/graphs node_vocabulary.pkl")
        sys.exit(1)
    
    vocab_file = sys.argv[1]
    max_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print(f"Using vocabulary file: {vocab_file}")
    print(f"Analyzing top {max_nodes} nodes")
    
    try:
        analyzer = DetailedNodeAnalyzer()
        df = analyzer.run_analysis(vocab_file, max_nodes=max_nodes)
        
        # Show final summary
        interpreted_count = df['neuronpedia_description'].notna().sum()
        print(f"\n✅ Analysis complete! Found interpretations for {interpreted_count} nodes.")
        print("Check the log directory for detailed results and summary report.")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()