"""
Convert attribution graphs from JSON format to NetworkX graphs for gnn-model-explainer
WITH PROPER SEMANTIC PROCESSING (like dataset.py and data_converter.py)
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
import tqdm
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticAttributionGraphConverter:
    """Convert attribution graphs with cross-layer transcoder filtering"""

    def __init__(self):
        # Simplified feature names - only cross-layer transcoder nodes
        self.feature_names = [
            'influence', 'activation', 'layer', 'ctx_idx', 'feature_id'
        ]
        self.feature_dims = len(self.feature_names)

    def _is_transcoder_node(self, node: Dict) -> bool:
        """Check if a node is a cross-layer transcoder node"""
        feature_type = node.get('feature_type', '')
        return feature_type == 'cross layer transcoder'

    def extract_node_features(self, node: Dict) -> List[float]:
        """Extract numeric features from a cross-layer transcoder node"""
        features = []

        # Basic numeric features - handle None values
        influence = node.get('influence', 0.0)
        features.append(float(influence if influence is not None else 0.0))

        activation = node.get('activation', 0.0)
        features.append(float(activation if activation is not None else 0.0))

        layer = node.get('layer', 0)
        features.append(float(layer if layer is not None else 0))

        ctx_idx = node.get('ctx_idx', 0)
        features.append(float(ctx_idx if ctx_idx is not None else 0))

        # Feature ID - preserve original without normalization since all nodes are transcoder
        feature_val = node.get('feature', 0)
        if feature_val is not None:
            features.append(float(feature_val))
        else:
            features.append(0.0)

        return features

    def _normalize_edge_weights_for_training(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize edge weights for stable training while preserving semantic meaning.
        (same as data_converter.py but using numpy)
        """
        if len(weights) == 0:
            return weights

        # Simple clamping approach: preserves relative magnitudes and is fast
        # Based on analysis: 99th percentile is ~2.0, so clamp at reasonable training range
        normalized_weights = np.clip(weights, -10.0, 10.0)

        # Add small epsilon to prevent exact zeros (some GNN layers sensitive to zeros)
        epsilon = 1e-8
        sign_mask = np.sign(normalized_weights)
        abs_weights = np.abs(normalized_weights)
        final_weights = sign_mask * np.clip(abs_weights, epsilon, None)

        return final_weights


def json_to_networkx_semantic(data: Dict[str, Any], label: int, converter: SemanticAttributionGraphConverter) -> nx.Graph:
    """
    Convert JSON attribution graph data to NetworkX graph with cross-layer transcoder filtering
    """
    G = nx.Graph()

    # Set graph-level label
    G.graph["label"] = label

    # Process nodes with transcoder filtering
    nodes_data = data.get("nodes", [])
    valid_nodes = []
    node_id_to_index = {}

    # Filter to keep only cross-layer transcoder nodes
    for node in nodes_data:
        if 'node_id' not in node or not converter._is_transcoder_node(node):
            continue
        valid_nodes.append(node)

    # Extract features for transcoder nodes
    for i, node_data in enumerate(valid_nodes):
        node_id = node_data['node_id']
        node_id_to_index[node_id] = i

        # Extract features from cross-layer transcoder nodes
        feat = np.array(converter.extract_node_features(node_data))

        # Check for NaN/Inf in node features
        if np.isnan(feat).any():
            logger.warning(f"Found NaN in node features, replacing with 0.0")
            feat = np.nan_to_num(feat, nan=0.0)

        if np.isinf(feat).any():
            logger.warning(
                f"Found Inf in node features, replacing with finite values")
            feat = np.nan_to_num(feat, posinf=100.0, neginf=-100.0)

        G.add_node(i, feat=feat)

    # Set feature dimension
    if len(valid_nodes) > 0:
        G.graph["feat_dim"] = len(G.nodes[0]["feat"])

    # Process edges with semantic filtering (same as data_converter.py)
    edges_data = data.get("links", data.get("edges", []))

    edge_weights = []
    skipped_edges = 0
    invalid_weight_edges = 0

    for edge_data in edges_data:
        source_id = edge_data.get("source")
        target_id = edge_data.get("target")

        # Skip edges that connect to non-transcoder nodes (which we filtered out)
        # This is crucial for consistency between nodes and edges
        if source_id not in node_id_to_index or target_id not in node_id_to_index:
            skipped_edges += 1
            continue

        source_idx = node_id_to_index[source_id]
        target_idx = node_id_to_index[target_id]

        # Extract and validate edge weight
        weight = edge_data.get("weight", 1.0)

        # Handle None weights
        if weight is None:
            weight = 1.0

        try:
            weight = float(weight)
        except (ValueError, TypeError):
            logger.warning(
                f"Non-numeric weight {weight} for edge {source_id}->{target_id}, setting to 1.0")
            weight = 1.0
            invalid_weight_edges += 1

        # Check for NaN/Inf weights
        if not (np.isfinite(weight) and not np.isnan(weight)):
            logger.warning(
                f"Invalid weight {weight} for edge {source_id}->{target_id}, setting to 1.0")
            weight = 1.0
            invalid_weight_edges += 1

        edge_weights.append(weight)
        G.add_edge(source_idx, target_idx, weight=weight)

    # Report edge filtering statistics
    total_original_edges = len(edges_data)
    valid_edges = len(edge_weights)
    logger.info(
        f"Edge processing: {total_original_edges} original → {valid_edges} valid ({skipped_edges} skipped, {invalid_weight_edges} had invalid weights)")

    # Normalize edge weights for training stability
    if edge_weights:
        normalized_weights = converter._normalize_edge_weights_for_training(
            np.array(edge_weights))

        # Update edge weights in graph
        for i, (u, v) in enumerate(G.edges()):
            G[u][v]['weight'] = normalized_weights[i]

    return G


def load_attribution_graphs_semantic(
    data_dir: str,
    cache_dir: str = None,
    max_files: int = None
) -> Tuple[List[nx.Graph], List[int]]:
    """
    Load attribution graphs with cross-layer transcoder filtering from directory structure
    """
    data_path = Path(data_dir)

    # Set up caching
    if cache_dir is None:
        cache_dir = data_path.parent / "gnn_explainer_cache_transcoder"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create cache filename
    cache_filename = f"transcoder_networkx_graphs_{data_path.name}"
    if max_files:
        cache_filename += f"_max{max_files}"
    cache_filename += ".pkl"
    cache_file = cache_path / cache_filename

    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached transcoder graphs from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data["graphs"], cached_data["labels"]
        except (EOFError, pickle.UnpicklingError, KeyError) as e:
            logger.warning(
                f"Cache file {cache_file} is corrupted ({e}), regenerating...")
            # Remove corrupted cache file
            cache_file.unlink()
        except Exception as e:
            logger.warning(
                f"Failed to load cache file {cache_file} ({e}), regenerating...")
            # Remove problematic cache file
            cache_file.unlink()

    logger.info(f"Loading transcoder-only graphs from {data_dir}")

    # Initialize converter
    converter = SemanticAttributionGraphConverter()

    graphs = []
    labels = []

    # Load benign graphs (label = 0)
    benign_dir = data_path / "benign"
    if benign_dir.exists():
        benign_files = list(benign_dir.glob("*.json"))
        if max_files:
            benign_files = random.sample(benign_files, max_files)

        logger.info(
            f"Loading {len(benign_files)} benign graphs with transcoder filtering")
        for file_path in tqdm.tqdm(benign_files, desc="Loading benign graphs"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                graph = json_to_networkx_semantic(
                    data, label=0, converter=converter)
                graphs.append(graph)
                labels.append(0)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

    # Load injected graphs (label = 1)
    injected_dir = data_path / "injected"
    if injected_dir.exists():
        injected_files = list(injected_dir.glob("*.json"))
        if max_files:
            injected_files = random.sample(injected_files, max_files)

        logger.info(
            f"Loading {len(injected_files)} injected graphs with transcoder filtering")
        for file_path in tqdm.tqdm(injected_files, desc="Loading injected graphs"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                graph = json_to_networkx_semantic(
                    data, label=1, converter=converter)
                graphs.append(graph)
                labels.append(1)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

    # Cache the results
    cache_data = {
        "graphs": graphs,
        "labels": labels,
        "data_dir": str(data_dir),
        "num_benign": sum(1 for l in labels if l == 0),
        "num_injected": sum(1 for l in labels if l == 1)
    }

    logger.info(f"Caching {len(graphs)} transcoder graphs to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    logger.info(f"Loaded {len(graphs)} transcoder graphs total:")
    logger.info(f"  Benign: {sum(1 for l in labels if l == 0)}")
    logger.info(f"  Injected: {sum(1 for l in labels if l == 1)}")

    return graphs, labels


def create_gnn_explainer_splits_semantic(
    data_dir: str,
    cache_dir: str = None,
    max_files: int = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph]]:
    """
    Create train/val/test splits for gnn-model-explainer with transcoder filtering
    """
    graphs, labels = load_attribution_graphs_semantic(
        data_dir, cache_dir, max_files
    )

    # Simple fixed split: 60% train, 20% val, 20% test
    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs, labels, test_size=(test_size+val_size), random_state=random_state, stratify=labels
    )

    # Split the remaining 40% into 20% val and 20% test
    val_graphs, test_graphs, val_labels, test_labels = train_test_split(
        temp_graphs, temp_labels, test_size=0.5, random_state=random_state, stratify=temp_labels
    )

    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_graphs)} graphs")
    logger.info(f"  Val: {len(val_graphs)} graphs")
    logger.info(f"  Test: {len(test_graphs)} graphs")

    return train_graphs, val_graphs, test_graphs


if __name__ == "__main__":
    # Test the semantic conversion
    data_dir = "/Users/samkouteili/rose/circuits/data_small"
    train_graphs, val_graphs, test_graphs = create_gnn_explainer_splits_semantic(
        data_dir, max_files=5, test_size=0.4, val_size=0.2  # Adjust for small dataset
    )

    if len(train_graphs) > 0:
        G = train_graphs[0]
        print(f"Sample semantic graph:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Label: {G.graph['label']}")
        print(f"  Feature dim: {G.graph.get('feat_dim', 'not set')}")
        print(f"  Features: {G.graph.get('feat_dim', 0)} dims")
        if G.number_of_nodes() > 0:
            print(f"  Sample node features: {G.nodes[0]['feat'][:5]}...")

        # Check edge weights
        if G.number_of_edges() > 0:
            weights = [G[u][v]['weight'] for u, v in list(G.edges())[:5]]
            print(f"  Sample edge weights: {weights}")
