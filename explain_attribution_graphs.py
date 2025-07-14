#!/usr/bin/env python3
"""
Run explainer on trained attribution graphs model

This is a simple wrapper that calls the existing explainer_main.py with the correct arguments
for attribution graphs models trained with train_attribution_graphs.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Explain attribution graphs model")

    # Model arguments (should match training script)
    parser.add_argument("--method", default="base",
                        help="Method: base, soft-assign")
    parser.add_argument("--hidden-dim", type=int,
                        default=64, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int,
                        default=64, help="Output dimension")
    parser.add_argument("--num-gc-layers", type=int,
                        default=3, help="Number of GC layers")

    # System arguments
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--cuda", default="0", help="CUDA device")
    parser.add_argument("--ckptdir", default="ckpt",
                        help="Model checkpoint directory")
    parser.add_argument("--logdir", default="log",
                        help="Tensorboard log directory")

    # Explainer arguments
    parser.add_argument("--graph-idx", type=int,
                        help="Specific graph index to explain (default: explain all)")
    parser.add_argument("--multigraph-class", type=int, choices=[0, 1],
                        help="Explain multiple graphs from same class: 0=benign, 1=injected")
    parser.add_argument("--num-graphs", type=int, default=5,
                        help="Number of graphs to explain from the class")
    parser.add_argument("--explainer-suffix", default="",
                        help="Suffix for explainer logs")

    args = parser.parse_args()

    # Build command line arguments for explainer_main.py
    explainer_cmd = [
        "python", "explainer_main.py",
        "--dataset", "attribution_graphs",
        "--method", args.method,
        "--ckptdir", args.ckptdir,
        "--logdir", args.logdir,
        "--hidden-dim", str(args.hidden_dim),
        "--output-dim", str(args.output_dim),
        "--num-gc-layers", str(args.num_gc_layers),
        "--graph-mode",  # We're doing graph classification
        "--cuda", args.cuda,
    ]

    if args.gpu:
        explainer_cmd.append("--gpu")

    if args.graph_idx is not None:
        explainer_cmd.extend(["--graph-idx", str(args.graph_idx)])
    
    if args.multigraph_class is not None:
        explainer_cmd.extend(["--multigraph-class", str(args.multigraph_class)])
    
    if args.explainer_suffix:
        explainer_cmd.extend(["--explainer-suffix", args.explainer_suffix])

    print("Running explainer with command:")
    print(" ".join(explainer_cmd))
    print()

    # Execute the command
    os.execvp("python", explainer_cmd)


if __name__ == "__main__":
    main()
