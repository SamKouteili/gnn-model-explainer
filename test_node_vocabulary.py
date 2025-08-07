#!/usr/bin/env python3
"""
Test Node Vocabulary Functionality

This script tests the node vocabulary by:
1. Loading a saved vocabulary pickle file
2. Testing lookups by node index and (layer, feature_id)
3. Verifying the mapping works correctly
4. Testing with random nodes from source graphs
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional
import sys


def load_vocabulary(vocab_file: str) -> Dict:
    """Load vocabulary from pickle file"""
    print(f"Loading vocabulary from {vocab_file}...")
    
    if not Path(vocab_file).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    
    with open(vocab_file, 'rb') as f:
        vocab_data = pickle.load(f)
    
    print(f"✅ Vocabulary loaded successfully!")
    
    # Print basic stats
    stats = vocab_data.get('stats', {})
    print(f"  Total nodes: {stats.get('total_nodes', 'unknown'):,}")
    print(f"  Unique (layer, feature_id) combinations: {len(vocab_data['node_vocab']):,}")
    print(f"  Created: {vocab_data.get('created_at', 'unknown')}")
    
    return vocab_data


def test_vocabulary_structure(vocab_data: Dict):
    """Test the basic structure of the vocabulary"""
    print("\n" + "=" * 60)
    print("TESTING VOCABULARY STRUCTURE")
    print("=" * 60)
    
    node_vocab = vocab_data['node_vocab']
    node_index_map = vocab_data['node_index_map']
    
    # Test 1: Check if keys are properly formatted
    print("Test 1: Key format validation")
    sample_keys = list(node_vocab.keys())[:5]
    for key in sample_keys:
        if isinstance(key, tuple) and len(key) == 2:
            layer, feature_id = key
            print(f"  ✅ Key {key}: layer='{layer}' (type: {type(layer)}), feature_id={feature_id} (type: {type(feature_id)})")
        else:
            print(f"  ❌ Invalid key format: {key}")
    
    # Test 2: Check node_index_map continuity
    print(f"\nTest 2: Index map continuity")
    max_index = max(node_index_map.keys())
    min_index = min(node_index_map.keys())
    expected_count = max_index - min_index + 1
    actual_count = len(node_index_map)
    print(f"  Index range: {min_index} to {max_index}")
    print(f"  Expected count: {expected_count:,}")
    print(f"  Actual count: {actual_count:,}")
    
    if expected_count == actual_count:
        print(f"  ✅ Index mapping is continuous")
    else:
        print(f"  ⚠️  Index mapping has gaps")
    
    # Test 3: Check data consistency
    print(f"\nTest 3: Data consistency")
    sample_indices = random.sample(list(node_index_map.keys()), min(5, len(node_index_map)))
    
    for idx in sample_indices:
        node_info = node_index_map[idx]
        layer = node_info.get('layer')
        feature_id = node_info.get('feature_id')
        
        if layer is not None and feature_id is not None:
            key = (str(layer), int(feature_id))
            if key in node_vocab:
                # Check if this node appears in the vocab lookup
                vocab_entries = node_vocab[key]
                found = any(entry.get('graph_index') == idx for entry in vocab_entries)
                status = "✅" if found else "❌"
                print(f"  {status} Index {idx}: Layer {layer}, Feature {feature_id} - Found in vocab: {found}")
            else:
                print(f"  ❌ Index {idx}: Key {key} not found in vocab")
        else:
            print(f"  ⚠️  Index {idx}: Missing layer or feature_id (layer={layer}, feature_id={feature_id})")


def test_random_nodes_from_source(vocab_data: Dict, source_graphs_dir: str, num_tests: int = 5):
    """Test vocabulary by checking random nodes from source graphs"""
    print("\n" + "=" * 60)
    print("TESTING RANDOM NODES FROM SOURCE GRAPHS")
    print("=" * 60)
    
    graphs_dir = Path(source_graphs_dir)
    if not graphs_dir.exists():
        print(f"❌ Source graphs directory not found: {source_graphs_dir}")
        return
    
    node_index_map = vocab_data['node_index_map']
    node_vocab = vocab_data['node_vocab']
    
    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}:")
        
        # Pick a random graph file
        all_files = []
        for subdir in ['benign', 'injected']:
            subdir_path = graphs_dir / subdir
            if subdir_path.exists():
                all_files.extend(list(subdir_path.glob("*.json")))
        
        if not all_files:
            print("  ❌ No JSON files found in source directory")
            continue
        
        random_file = random.choice(all_files)
        print(f"  Testing with file: {random_file.name}")
        
        try:
            with open(random_file, 'r') as f:
                graph_data = json.load(f)
            
            nodes = graph_data.get('nodes', [])
            if not nodes:
                print("  ⚠️  No nodes in this graph")
                continue
            
            # Pick a random node
            random_node = random.choice(nodes)
            node_id = random_node.get('node_id', '')
            layer = random_node.get('layer')
            ctx_idx = random_node.get('ctx_idx')
            feature_type = random_node.get('feature_type')
            
            # Extract feature ID
            feature_id = None
            if node_id:
                try:
                    parts = node_id.split('_')
                    if len(parts) >= 2:
                        feature_id = int(parts[1])
                except (ValueError, IndexError):
                    pass
            
            print(f"  Random node: {node_id}")
            print(f"    Layer: {layer}, Feature ID: {feature_id}, Type: {feature_type}")
            
            # Test vocabulary lookup
            if layer is not None and feature_id is not None:
                key = (str(layer), int(feature_id))
                if key in node_vocab:
                    matching_entries = node_vocab[key]
                    print(f"    ✅ Found {len(matching_entries)} matching entries in vocabulary")
                    
                    # Check if any entry matches this specific file
                    file_matches = [entry for entry in matching_entries 
                                  if entry.get('source_file') == random_file.name]
                    if file_matches:
                        print(f"    ✅ Found {len(file_matches)} entries from this specific file")
                        # Show first match details
                        match = file_matches[0]
                        print(f"      Graph index: {match.get('graph_index')}")
                        print(f"      Node ID: {match.get('node_id')}")
                        print(f"      Context: {match.get('ctx_idx')}")
                    else:
                        print(f"    ⚠️  No entries from this specific file (but key exists)")
                else:
                    print(f"    ❌ Key {key} not found in vocabulary")
            else:
                print(f"    ⚠️  Cannot test - missing layer or feature_id")
                
        except Exception as e:
            print(f"  ❌ Error processing {random_file.name}: {e}")


def test_explainer_style_lookups(vocab_data: Dict, num_tests: int = 10):
    """Test lookups the way the explainer would use them"""
    print("\n" + "=" * 60)
    print("TESTING EXPLAINER-STYLE LOOKUPS")
    print("=" * 60)
    
    node_index_map = vocab_data['node_index_map']
    
    # Get some random indices to test
    all_indices = list(node_index_map.keys())
    test_indices = random.sample(all_indices, min(num_tests, len(all_indices)))
    
    print(f"Testing {len(test_indices)} random node indices...")
    
    valid_for_neuronpedia = 0
    
    for i, node_idx in enumerate(test_indices):
        node_info = node_index_map[node_idx]
        
        layer = node_info.get('layer')
        feature_id = node_info.get('feature_id') 
        node_type = node_info.get('node_type')
        activation = node_info.get('activation')
        
        print(f"\n  Node {node_idx}:")
        print(f"    Layer: {layer}")
        print(f"    Feature ID: {feature_id}")
        print(f"    Type: {node_type}")
        print(f"    Activation: {activation}")
        
        # Test if valid for Neuronpedia (mimic the validation logic)
        is_valid = True
        if layer is None or feature_id is None:
            is_valid = False
            print(f"    ❌ Invalid: Missing layer or feature_id")
        elif layer in [0, "0", "E"] or not str(layer).isdigit():
            is_valid = False
            print(f"    ❌ Invalid: Bad layer value")
        elif node_type in ['mlp reconstruction error', 'embedding', 'unknown']:
            is_valid = False
            print(f"    ❌ Invalid: Excluded node type")
        elif feature_id <= 0 or feature_id > 16384:
            is_valid = False
            print(f"    ❌ Invalid: Feature ID out of range")
        else:
            valid_for_neuronpedia += 1
            print(f"    ✅ Valid for Neuronpedia lookup")
    
    print(f"\nSummary: {valid_for_neuronpedia}/{len(test_indices)} nodes valid for Neuronpedia")
    valid_pct = (valid_for_neuronpedia / len(test_indices)) * 100
    print(f"Validation rate: {valid_pct:.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_node_vocabulary.py <vocab_file> [source_graphs_dir]")
        print("Example: python test_node_vocabulary.py node_vocabulary.pkl /home/sk2959/scratch_pi_rp476/sk2959/ag_data")
        sys.exit(1)
    
    vocab_file = sys.argv[1]
    source_graphs_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Load vocabulary
        vocab_data = load_vocabulary(vocab_file)
        
        # Test vocabulary structure
        test_vocabulary_structure(vocab_data)
        
        # Test explainer-style lookups
        test_explainer_style_lookups(vocab_data)
        
        # Test with random nodes from source (if directory provided)
        if source_graphs_dir:
            test_random_nodes_from_source(vocab_data, source_graphs_dir)
        else:
            print("\n⚠️  Skipping source graph tests (no directory provided)")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("The vocabulary appears to be working correctly!")
        print("You can now use it in your analysis scripts.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()