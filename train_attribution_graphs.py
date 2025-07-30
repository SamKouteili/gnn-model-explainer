#!/usr/bin/env python3
"""
Train GNN model on attribution graphs using gnn-model-explainer
"""

from train import prepare_data, train, evaluate
from convert_attribution_graphs_semantic import create_gnn_explainer_splits_semantic
import configs
import models
from tensorboardX import SummaryWriter
import torch
import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def train_attribution_graphs(args):
    """Train GNN on attribution graphs"""

    # Load and split data
    print("Loading attribution graphs with semantic processing...")
    train_graphs, val_graphs, test_graphs = create_gnn_explainer_splits_semantic(
        args.data_dir,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        test_size=args.test_ratio,
        val_size=0.1,
        random_state=42
    )

    # Prepare data loaders
    print("Preparing data loaders...")

    # Calculate actual max nodes from our graphs
    # all_graphs = train_graphs + val_graphs + test_graphs
    # actual_max_nodes = max([G.number_of_nodes() for G in all_graphs])
    # print(f"Actual max nodes in dataset: {actual_max_nodes}")

    # # Use the larger of args.max_nodes or actual_max_nodes
    # max_num_nodes = max(args.max_nodes, actual_max_nodes)
    # print(f"Using max_num_nodes: {max_num_nodes}")

    train_dataset, val_dataset, test_dataset, _, input_dim, assign_input_dim = prepare_data(
        # All graphs for max_num_nodes calculation
        (train_graphs + val_graphs + test_graphs),
        args,
        test_graphs=test_graphs,
        max_nodes=args.max_nodes
    )

    # Override the datasets to use our splits
    # from utils.graph_utils import GraphSampler

    # train_dataset = torch.utils.data.DataLoader(
    #     GraphSampler(train_graphs, normalize=False,
    #                  max_num_nodes=args.max_nodes, features=args.feature_type),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    # )

    # val_dataset = torch.utils.data.DataLoader(
    #     GraphSampler(val_graphs, normalize=False,
    #                  max_num_nodes=args.max_nodes, features=args.feature_type),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    # )

    # test_dataset = torch.utils.data.DataLoader(
    #     GraphSampler(test_graphs, normalize=False,
    #                  max_num_nodes=args.max_nodes, features=args.feature_type),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    # )

    print(f"Dataset info:")
    print(f"  Max nodes: {args.max_nodes}")
    print(f"  Input dim: {input_dim}")
    print(f"  Assign input dim: {assign_input_dim}")
    print(f"  Train batches: {len(train_dataset)}")
    print(f"  Val batches: {len(val_dataset)}")
    print(f"  Test batches: {len(test_dataset)}")

    # Create model
    print("Creating model...")
    if args.method == "soft-assign":
        model = models.SoftPoolingGcnEncoder(
            args.max_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        )
    else:
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        )

    if args.gpu:
        model = model.cuda()

    print(f"Model: {model}")

    # Set up logging
    writer = None
    if args.logdir:
        from utils.io_utils import gen_prefix
        log_path = os.path.join(args.logdir, gen_prefix(args))
        writer = SummaryWriter(log_path)

    # Train
    print("Starting training...")
    train(
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )

    # Final evaluation
    print("Final evaluation...")
    evaluate(test_dataset, model, args, "Final Test")

    if writer:
        writer.close()

    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN on attribution graphs")

    # Data arguments
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing benign/ and injected/ subdirs")
    parser.add_argument("--cache-dir",
                        help="Cache directory for processed graphs")
    parser.add_argument("--max-files", type=int,
                        help="Maximum files per category (for testing)")

    # Model arguments
    parser.add_argument("--method", default="base",
                        help="Method: base, soft-assign")
    parser.add_argument("--hidden-dim", type=int,
                        default=64, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int,
                        default=64, help="Output dimension")
    parser.add_argument("--num-classes", type=int,
                        default=2, help="Number of classes")
    parser.add_argument("--num-gc-layers", type=int,
                        default=3, help="Number of GC layers")
    parser.add_argument("--dropout", type=float,
                        default=0.1, help="Dropout rate")
    parser.add_argument("--bn", action="store_true",
                        help="Use batch normalization")

    # Training arguments
    parser.add_argument("--batch-size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float,
                        default=0.005, help="Weight decay")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="Gradient clipping")

    # Data processing
    parser.add_argument("--max-nodes", type=int, default=1000,
                        help="Maximum number of nodes")
    parser.add_argument("--feature-type", default="default",
                        help="Feature type: default, id, deg")
    parser.add_argument("--test-ratio", type=float,
                        default=0.2, help="Test set ratio")
    parser.add_argument("--train-ratio", type=float,
                        default=0.8, help="Train set ratio")

    # System arguments
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--cuda", default="0", help="CUDA device")
    parser.add_argument("--num-workers", type=int,
                        default=1, help="Number of workers")
    parser.add_argument("--logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", help="Model checkpoint directory")

    # Soft pooling arguments
    parser.add_argument("--assign-ratio", type=float,
                        default=0.1, help="Assignment ratio")
    parser.add_argument("--num-pool", type=int, default=1,
                        help="Number of pooling layers")
    parser.add_argument("--linkpred", action="store_true",
                        help="Use link prediction")

    args = parser.parse_args()

    # Set defaults
    if args.logdir is None:
        args.logdir = "log"
    if args.ckptdir is None:
        args.ckptdir = "ckpt"

    # GPU setup
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        print(f"Using CUDA device: {args.cuda}")
    else:
        print("Using CPU")

    # Create directories
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.ckptdir, exist_ok=True)

    # Set additional required attributes for compatibility
    args.num_epochs = args.epochs
    args.bias = True
    args.name_suffix = ""
    args.bmname = None
    args.dataset = "attribution_graphs"
    args.pkl_fname = None

    # Train
    train_attribution_graphs(args)


if __name__ == "__main__":
    main()
