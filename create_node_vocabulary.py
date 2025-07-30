#!/usr/bin/env python3
"""
Create Node Vocabulary from Source Attribution Graphs

This script processes all source attribution graphs and creates a vocabulary
mapping node indices to semantic information. The vocabulary is saved as a
pickle file for fast loading in subsequent analyses.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
from collections import defaultdict


def extract_feature_id_from_node_id(node_id: str) -> Optional[int]:
    """Extract feature ID from node_id format like '1_365_38'"""
    if not node_id:
        return None
    try:
        parts = node_id.split('_')
        if len(parts) >= 2:
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def create_node_vocabulary(source_graphs_dir: str) -> Tuple[Dict, Dict]:
    """Create a vocabulary of all nodes from source graphs for lookup"""
    print("=" * 80)
    print("CREATING NODE VOCABULARY FROM SOURCE ATTRIBUTION GRAPHS")
    print("=" * 80)
    
    graphs_dir = Path(source_graphs_dir)
    if not graphs_dir.exists():
        raise FileNotFoundError(f"Source graphs directory not found: {source_graphs_dir}")
    
    node_vocab = {}  # Maps (layer, feature_id) -> [node_info, ...]
    node_index_map = {}  # Maps sequential index -> node_info
    current_index = 0
    
    # Statistics tracking
    stats = {
        'total_files': 0,
        'total_nodes': 0,
        'nodes_by_type': defaultdict(int),
        'nodes_by_layer': defaultdict(int),
        'files_processed': [],
        'files_failed': []
    }
    
    # Process all graph files
    for subdir in ['benign', 'injected']:
        subdir_path = graphs_dir / subdir
        if not subdir_path.exists():
            print(f"Warning: {subdir} directory not found, skipping...")
            continue
            
        print(f"\nProcessing {subdir} graphs...")
        json_files = sorted(subdir_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {subdir}/")
        
        for json_file in json_files:
            stats['total_files'] += 1
            try:
                print(f"  Processing {json_file.name}...", end="")
                
                with open(json_file, 'r') as f:
                    graph_data = json.load(f)
                
                nodes = graph_data.get('nodes', [])
                file_node_count = 0
                
                for node in nodes:
                    # Extract key information
                    layer = node.get('layer')
                    feature_id = extract_feature_id_from_node_id(node.get('node_id', ''))
                    ctx_idx = node.get('ctx_idx')
                    feature_type = node.get('feature_type', 'unknown')
                    activation = node.get('activation')
                    influence = node.get('influence')
                    node_id = node.get('node_id')
                    
                    node_info = {
                        'layer': layer,
                        'feature_id': feature_id,
                        'ctx_idx': ctx_idx,
                        'node_type': feature_type,
                        'activation': activation,
                        'influence': influence,
                        'node_id': node_id,
                        'source_file': json_file.name,
                        'source_type': subdir,
                        'graph_index': current_index  # Track where this node appears in sequence
                    }
                    
                    # Add to vocabulary with multiple keys for flexible lookup
                    if layer is not None and feature_id is not None:
                        # Key by (layer, feature_id) for primary lookup
                        key = (str(layer), int(feature_id))
                        if key not in node_vocab:
                            node_vocab[key] = []
                        node_vocab[key].append(node_info)
                    
                    # Map by sequential index (critical for explainer node mapping)
                    node_index_map[current_index] = node_info
                    
                    # Update statistics
                    stats['total_nodes'] += 1
                    stats['nodes_by_type'][feature_type] += 1
                    stats['nodes_by_layer'][str(layer)] += 1
                    
                    current_index += 1
                    file_node_count += 1
                
                print(f" {file_node_count} nodes")
                stats['files_processed'].append(json_file.name)
                
            except Exception as e:
                print(f" ERROR: {e}")
                stats['files_failed'].append((json_file.name, str(e)))
                continue
    
    # Print statistics
    print("\n" + "=" * 80)
    print("VOCABULARY CREATION STATISTICS")
    print("=" * 80)
    print(f"Files processed successfully: {len(stats['files_processed'])}")
    print(f"Files failed: {len(stats['files_failed'])}")
    print(f"Total nodes indexed: {stats['total_nodes']:,}")
    print(f"Unique (layer, feature_id) combinations: {len(node_vocab):,}")
    
    print(f"\nNodes by type:")
    for node_type, count in sorted(stats['nodes_by_type'].items()):
        pct = (count / stats['total_nodes']) * 100
        print(f"  {node_type}: {count:,} ({pct:.1f}%)")
    
    print(f"\nNodes by layer:")
    layer_counts = [(k, v) for k, v in stats['nodes_by_layer'].items()]
    layer_counts.sort(key=lambda x: (x[0] if x[0].isdigit() else 'zzz', x[0]))
    for layer, count in layer_counts:
        pct = (count / stats['total_nodes']) * 100
        print(f"  Layer {layer}: {count:,} ({pct:.1f}%)")
    
    if stats['files_failed']:
        print(f"\nFailed files:")
        for filename, error in stats['files_failed']:
            print(f"  {filename}: {error}")
    
    return node_vocab, node_index_map, stats


def save_vocabulary(node_vocab: Dict, node_index_map: Dict, stats: Dict, output_path: str):
    """Save vocabulary to pickle file"""
    print(f"\nSaving vocabulary to {output_path}...")
    
    vocab_data = {
        'node_vocab': node_vocab,
        'node_index_map': node_index_map,
        'stats': stats,
        'created_at': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    # Check file size
    file_size = Path(output_path).stat().st_size
    size_mb = file_size / (1024 * 1024)
    print(f"Vocabulary saved successfully! File size: {size_mb:.1f} MB")


def main():
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python create_node_vocabulary.py <source_graphs_dir> [output_file]")
        print("Example: python create_node_vocabulary.py /home/sk2959/scratch_pi_rp476/sk2959/ag_data")
        sys.exit(1)
    
    source_graphs_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "node_vocabulary.pkl"
    
    print(f"Source graphs directory: {source_graphs_dir}")
    print(f"Output file: {output_file}")
    
    try:
        # Create vocabulary
        node_vocab, node_index_map, stats = create_node_vocabulary(source_graphs_dir)
        
        # Save to pickle file
        save_vocabulary(node_vocab, node_index_map, stats, output_file)
        
        print("\n✅ Vocabulary creation complete!")
        print(f"Use this vocabulary file in your analysis scripts: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error creating vocabulary: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()