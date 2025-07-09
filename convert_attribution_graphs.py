"""
Convert attribution graphs from JSON format to NetworkX graphs for gnn-model-explainer
"""

import json
import os
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a single JSON file containing attribution graph data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def json_to_networkx(data: Dict[str, Any], label: int) -> nx.Graph:
    """
    Convert JSON attribution graph data to NetworkX graph
    
    Args:
        data: Dictionary containing nodes, edges, and other graph data
        label: Graph-level label (0 for benign, 1 for injected)
    
    Returns:
        NetworkX graph with proper node features and graph label
    """
    G = nx.Graph()
    
    # Set graph-level label
    G.graph["label"] = label
    
    # Add nodes with features
    nodes_data = data.get("nodes", [])
    for i, node_data in enumerate(nodes_data):
        # Extract node features based on the actual JSON structure
        # Handle string values and None values safely
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        feat = np.array([
            safe_float(node_data.get("feature", 0)),           # feature ID
            safe_float(node_data.get("layer", 0)),             # layer number
            safe_float(node_data.get("ctx_idx", i)),           # context index
            safe_float(node_data.get("token_prob", 0.0)),      # token probability
            safe_float(node_data.get("is_target_logit", False)), # is target logit
            safe_float(node_data.get("run_idx", 0)),           # run index
            safe_float(node_data.get("reverse_ctx_idx", 0)),   # reverse context index
            safe_float(node_data.get("influence", 0.0)),       # influence score
            safe_float(node_data.get("activation", 0.0)),      # activation value
        ])
        
        G.add_node(i, feat=feat)
    
    # Set feature dimension
    if len(nodes_data) > 0:
        G.graph["feat_dim"] = len(G.nodes[0]["feat"])
    
    # Add edges (stored under 'links' key)
    edges_data = data.get("links", data.get("edges", []))
    
    # Create a mapping from node_id to index
    node_id_to_index = {}
    for i, node_data in enumerate(nodes_data):
        node_id_to_index[node_data["node_id"]] = i
    
    for edge_data in edges_data:
        source_id = edge_data.get("source", edge_data.get("from"))
        target_id = edge_data.get("target", edge_data.get("to"))
        weight = edge_data.get("weight", edge_data.get("attribution", 1.0))
        
        # Map node IDs to indices
        source_idx = node_id_to_index.get(source_id)
        target_idx = node_id_to_index.get(target_id)
        
        # Only add edge if both nodes exist
        if source_idx is not None and target_idx is not None:
            G.add_edge(source_idx, target_idx, weight=weight)
    
    return G

def load_attribution_graphs_from_directory(
    data_dir: str,
    cache_dir: str = None,
    max_files: int = None
) -> Tuple[List[nx.Graph], List[int]]:
    """
    Load attribution graphs from directory structure similar to dataset.py
    
    Args:
        data_dir: Directory containing benign/ and injected/ subdirectories
        cache_dir: Directory for caching processed graphs
        max_files: Maximum number of files to process per category
    
    Returns:
        Tuple of (graphs_list, labels_list)
    """
    data_path = Path(data_dir)
    
    # Set up caching
    if cache_dir is None:
        cache_dir = data_path.parent / "gnn_explainer_cache"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename based on data directory and max_files
    cache_filename = f"networkx_graphs_{data_path.name}"
    if max_files:
        cache_filename += f"_max{max_files}"
    cache_filename += ".pkl"
    cache_file = cache_path / cache_filename
    
    logger.info(f"Cache file path: {cache_file}")
    
    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached graphs from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data["graphs"], cached_data["labels"]
    
    logger.info(f"Loading graphs from {data_dir}")
    
    graphs = []
    labels = []
    
    # Load benign graphs (label = 0)
    benign_dir = data_path / "benign"
    if benign_dir.exists():
        benign_files = list(benign_dir.glob("*.json"))
        if max_files:
            benign_files = benign_files[:max_files]
        
        logger.info(f"Loading {len(benign_files)} benign graphs")
        for file_path in benign_files:
            try:
                data = load_json_file(file_path)
                graph = json_to_networkx(data, label=0)
                graphs.append(graph)
                labels.append(0)
                logger.info(f"Successfully loaded {file_path}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Load injected graphs (label = 1)
    injected_dir = data_path / "injected"
    if injected_dir.exists():
        injected_files = list(injected_dir.glob("*.json"))
        if max_files:
            injected_files = injected_files[:max_files]
        
        logger.info(f"Loading {len(injected_files)} injected graphs")
        for file_path in injected_files:
            try:
                data = load_json_file(file_path)
                graph = json_to_networkx(data, label=1)
                graphs.append(graph)
                labels.append(1)
                logger.info(f"Successfully loaded {file_path}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Cache the results
    cache_data = {
        "graphs": graphs,
        "labels": labels,
        "data_dir": str(data_dir),
        "num_benign": sum(1 for l in labels if l == 0),
        "num_injected": sum(1 for l in labels if l == 1)
    }
    
    logger.info(f"Caching {len(graphs)} graphs to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    logger.info(f"Loaded {len(graphs)} graphs total:")
    logger.info(f"  Benign: {sum(1 for l in labels if l == 0)}")
    logger.info(f"  Injected: {sum(1 for l in labels if l == 1)}")
    
    return graphs, labels

def create_gnn_explainer_splits(
    data_dir: str,
    cache_dir: str = None,
    max_files: int = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph]]:
    """
    Create train/val/test splits for gnn-model-explainer
    
    Args:
        data_dir: Directory containing the data
        cache_dir: Cache directory
        max_files: Maximum files per category
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
    
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    graphs, labels = load_attribution_graphs_from_directory(
        data_dir, cache_dir, max_files
    )
    
    # Create train/test split
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Create train/val split
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_graphs, train_labels, test_size=val_size_adjusted, 
        random_state=random_state, stratify=train_labels
    )
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_graphs)} graphs")
    logger.info(f"  Val: {len(val_graphs)} graphs") 
    logger.info(f"  Test: {len(test_graphs)} graphs")
    
    return train_graphs, val_graphs, test_graphs

if __name__ == "__main__":
    # Example usage
    data_dir = "../circuit-tracer/data_small"
    train_graphs, val_graphs, test_graphs = create_gnn_explainer_splits(
        data_dir, max_files=50  # Small test
    )
    
    # Test a single graph
    if len(train_graphs) > 0:
        G = train_graphs[0]
        print(f"Sample graph:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Label: {G.graph['label']}")
        print(f"  Feature dim: {G.graph.get('feat_dim', 'not set')}")
        if G.number_of_nodes() > 0:
            print(f"  Sample node features: {G.nodes[0]['feat']}")