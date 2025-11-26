#!/usr/bin/env python3
import os, json, glob, argparse
from typing import Dict, List, Tuple
import torch

# -------- feature selection (edit as needed) --------
NODE_FEATURE_KEYS = ["activation", "influence"]   # numeric features to learn from
ADD_IS_ACTIVE_FLAG = True                         # adds a 0/1 flag per node

def load_master_topology(master_path: str) -> Tuple[Dict[str, int], List[str]]:
    """Load master_topology.json and return id2idx + idx2id."""
    with open(master_path, "r") as f:
        topo = json.load(f)
    nodes: List[str] = topo["nodes"]
    id2idx = {nid: i for i, nid in enumerate(nodes)}
    idx2id = nodes[:]  # same order
    return id2idx, idx2id

def build_plain_sample(
    json_path: str,
    id2idx: Dict[str, int],
    label: int,
    node_feature_keys = NODE_FEATURE_KEYS,
    add_is_active_flag: bool = ADD_IS_ACTIVE_FLAG,
    use_float16: bool = False,
) -> dict:
    """Convert one JSON into a plain dict of tensors (no PyG classes)."""
    with open(json_path, "r") as f:
        g = json.load(f)

    N = len(id2idx)
    F = len(node_feature_keys) + (1 if add_is_active_flag else 0)

    dtype = torch.float16 if use_float16 else torch.float32

    # --- Node features: dense [N, F], zero by default
    x = torch.zeros((N, F), dtype=dtype)

    # Fill present nodes
    present = set()
    for n in g.get("nodes", []):
        nid = n["node_id"]
        if nid not in id2idx:
            continue
        i = id2idx[nid]
        present.add(i)

        vals = []
        for k in node_feature_keys:
            v = n.get(k, 0.0)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            try:
                v = float(v)
            except Exception:
                v = 0.0
            vals.append(v)
        if add_is_active_flag:
            vals.append(1.0)
        x[i] = torch.tensor(vals, dtype=dtype)

    # --- Edges for this sample
    src, dst, w = [], [], []
    for e in g.get("links", []):
        s_id, t_id = e["source"], e["target"]
        if s_id in id2idx and t_id in id2idx:
            src.append(id2idx[s_id]); dst.append(id2idx[t_id])
            w.append(float(e.get("weight", 1.0)))

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=dtype)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(w, dtype=dtype).unsqueeze(-1)  # [E,1]

    sample = {
        "x": x,                               # [N,F] float{16,32}
        "edge_index": edge_index,             # [2,E] long
        "edge_attr": edge_attr,               # [E,1] float{16,32}
        "y": torch.tensor([label], dtype=torch.long),
        "num_active_nodes": torch.tensor([len(present)], dtype=torch.long),
        "path": json_path,                    # keep original path for reference
    }
    return sample

def discover_files(root: str) -> List[tuple]:
    """Yield (path, label) for all JSONs under benign/ and injected/."""
    pairs = []
    for sub, label in (("benign", 0), ("injected", 1)):
        subdir = os.path.join(root, sub)
        files = sorted(glob.glob(os.path.join(subdir, "*.json")))
        for p in files:
            pairs.append((p, label))
    return pairs

def main():
    ap = argparse.ArgumentParser(description="Convert attribution JSONs to plain-tensor dataset (no PyG pickling).")
    ap.add_argument("data_dir", type=str, help="Directory containing benign/ and injected/ subfolders")
    ap.add_argument("--master", type=str, default=None, help="Path to master_topology.json (default: <data_dir>/master_topology.json)")
    ap.add_argument("--out", type=str, default=None, help="Output .pt path (default: <data_dir>/data.pt)")
    ap.add_argument("--float16", action="store_true", help="Store x/edge_attr as float16 to reduce size")
    args = ap.parse_args()

    master_path = args.master or os.path.join(args.data_dir, "master_topology.json")
    out_path = args.out or os.path.join(args.data_dir, "data.pt")

    id2idx, idx2id = load_master_topology(master_path)
    print(f"[+] Master: {len(idx2id)} nodes")

    files = discover_files(args.data_dir)
    print(f"[+] Found {len(files)} JSON samples")

    graphs_plain: List[dict] = []
    for i, (p, lbl) in enumerate(files, 1):
        sample = build_plain_sample(
            p, id2idx, lbl,
            node_feature_keys=NODE_FEATURE_KEYS,
            add_is_active_flag=ADD_IS_ACTIVE_FLAG,
            use_float16=args.float16,
        )
        graphs_plain.append(sample)
        if i % 50 == 0:
            print(f"  … processed {i}/{len(files)}")

    bundle = {
        "graphs": graphs_plain,          # list[dict of tensors]
        "id2idx": id2idx,                # global mapping (dict[str,int])
        "idx2id": idx2id,                # list[str]
        "node_feature_keys": NODE_FEATURE_KEYS,
        "add_is_active_flag": ADD_IS_ACTIVE_FLAG,
        "dtype": "float16" if args.float16 else "float32",
    }
    torch.save(bundle, out_path)
    print(f"[✓] Saved {len(graphs_plain)} samples to {out_path}")

if __name__ == "__main__":
    main()
