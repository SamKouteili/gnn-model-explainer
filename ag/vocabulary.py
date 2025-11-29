import os
import glob
import json
import argparse
from collections import defaultdict

def build_master_topology(data_dir):
    """
    Builds a global node/edge topology by taking the union across all graphs
    under data_dir/{benign,injected}.
    """

    node_ids = set()
    edges = set()

    for sub in ("benign", "injected"):
        subdir = os.path.join(data_dir, sub)
        if not os.path.exists(subdir):
            continue

        json_files = glob.glob(os.path.join(subdir, "*.json"))
        print(f"[+] Found {len(json_files)} JSON files in {subdir}")

        for path in json_files:
            try:
                with open(path, "r") as f:
                    g = json.load(f)
                
                clt_in_graph = set()


                for n in g.get("nodes", []):
                    if n["feature_type"] == "cross layer transcoder":
                        clt_in_graph.add(n["node_id"])

                for e in g.get("links", []):
                    if e["weight"] > 1. or e["weight"] < -1.: 
                        if e["source"] in clt_in_graph and e["target"] in clt_in_graph:
                            edges.add((e["source"], e["target"]))
                
                node_ids |= clt_in_graph
                print(f"-- {path} processed, edges: {len(edges)}")
            except Exception as ex:
                print(f"[!] Failed to read {path}: {ex}")

    # Freeze consistent ordering
    node_ids = sorted(node_ids)
    edges = sorted(edges)

    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_edges = [(id2idx[s], id2idx[t]) for s, t in edges]

    print(f"\n[+] Total unique nodes: {len(node_ids)}")
    print(f"[+] Total unique edges: {len(edges)}")

    # Save to disk for reproducibility
    out_path = os.path.join(data_dir, "vocabuary.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "num_nodes": len(node_ids),
                "num_edges": len(edges),
                "nodes": node_ids,
                "edges": [{"source": s, "target": t} for s, t in edges],
            },
            f,
            indent=2,
        )

    print(f"[✓] Saved master topology to {out_path}")
    return id2idx, idx_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a master transcoder topology from attribution graphs."
    )
    parser.add_argument("data_dir", type=str, help="Path containing benign/ and injected/ folders")
    args = parser.parse_args()

    build_master_topology(args.data_dir)
