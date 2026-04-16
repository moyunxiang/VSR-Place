#!/usr/bin/env python3
"""Parse ISPD2005 Bookshelf benchmarks into PyG pickle files.

Downloads ISPD2005 data if not present, parses each circuit,
and saves as graph{idx}.pickle + output{idx}.pickle.

Usage:
    python scripts/parse_ispd2005.py
    python scripts/parse_ispd2005.py --benchmark-dir /path/to/ispd2005
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHIPDIFFUSION_ROOT = PROJECT_ROOT / "third_party" / "chipdiffusion"

# Chip sizes for ISPD2005 circuits (from ChipDiffusion notebooks)
# Format: [x_start, y_start, x_end, y_end] in units/1000
ISPD_CHIP_SIZES = {
    0: [0.459, 0.459, 0.459 + 10692 / 1000, 0.459 + 12 * 890 / 1000],   # adaptec1
    1: [0.609, 0.616, 0.609 + 14054 / 1000, 0.616 + 12 * 1170 / 1000],  # adaptec2
    2: [0.036, 0.058, 0.036 + 23190 / 1000, 23386 / 1000],               # adaptec3
    3: [0.036, 0.058, 0.036 + 23190 / 1000, 23386 / 1000],               # adaptec4
    4: [0.459, 0.459, 0.459 + 10692 / 1000, 11139 / 1000],               # bigblue1
    5: [0.036, 0.076, 0.036 + 18690 / 1000, 18868 / 1000],               # bigblue2
    6: [0.036, 0.076, 0.036 + 27690 / 1000, 27868 / 1000],               # bigblue3
    7: [0.036, 0.058, 0.036 + 32190 / 1000, 32386 / 1000],               # bigblue4
}

ISPD_NAMES = [f"adaptec{i+1}" for i in range(4)] + [f"bigblue{i+1}" for i in range(4)]
SCALING_UNITS = 1000


class Pin:
    def __init__(self, id, obj_name, offset):
        self.obj_name = obj_name
        self.id = id
        self.offset = offset


def flip_stack(data):
    """Double edges: (u->v) becomes (u->v, v->u)."""
    return torch.concatenate((data, torch.flip(data, dims=(1,))), dim=0)


def parse_bookshelf(nodes_path, nets_path, pl_path):
    """Parse ISPD2005 Bookshelf format into PyG Data + placement tensor.

    Args:
        nodes_path: Path to .nodes file (component sizes).
        nets_path: Path to .nets file (netlist connections).
        pl_path: Path to .pl file (placement coordinates).

    Returns:
        Tuple of (x, cond) where:
            x: (V, 2) tensor of placement coordinates (scaled).
            cond: PyG Data object with graph structure.
    """
    # Parse .pl (placement)
    with open(pl_path) as f:
        pl_lines = f.readlines()
    pl_pattern = re.compile(
        r"^\s*(\S+)\s+(\d+(?:\.\d*)?)\s+(\d+(?:\.\d+)?)\s*:\s*(F?N|S|E|W)\s*(/FIXED)?\s*$"
    )
    placement = {"names": [], "positions": [], "is_fixed": []}
    for line in pl_lines:
        m = re.match(pl_pattern, line)
        if m:
            placement["names"].append(m[1])
            placement["positions"].append((float(m[2]), float(m[3])))
            placement["is_fixed"].append(m[5] is not None)

    # Parse .nodes (sizes)
    with open(nodes_path) as f:
        nodes_lines = f.readlines()
    node_pattern = re.compile(r"^\s*(\S+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d*)?)\s+(terminal)?$")
    node_sizes = {}
    node_macros = {}
    for line in nodes_lines:
        m = re.match(node_pattern, line)
        if m:
            node_sizes[m[1]] = (float(m[2]), float(m[3]))
            node_macros[m[1]] = m[4] is not None

    # Parse .nets (netlist)
    with open(nets_path) as f:
        nets_lines = f.readlines()
    header_pattern = re.compile(r"^\s*NetDegree\s*:\s*(\d+)\s+(\S+)\s*$")
    pin_pattern = re.compile(r"^\s*(\S+)\s+(I|O)\s*:\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$")

    nets = []
    i = 0
    while i < len(nets_lines):
        hm = re.match(header_pattern, nets_lines[i])
        i += 1
        if not hm:
            continue
        degree = int(hm[1])
        if degree < 2:
            i += degree
            continue
        net = {"inputs": []}
        for _ in range(degree):
            pm = re.match(pin_pattern, nets_lines[i])
            i += 1
            if not pm:
                continue
            pin = Pin(id=i, obj_name=pm[1], offset=(float(pm[3]), float(pm[4])))
            if pm[2] == "O" and "output" not in net:
                net["output"] = pin
            else:
                net["inputs"].append(pin)
        if "output" not in net:
            net["output"] = net["inputs"].pop()
        nets.append(net)

    # Build tensors
    x = torch.tensor(placement["positions"])
    name_to_idx = {}
    cond_x = []
    is_macros_list = []
    for idx, name in enumerate(placement["names"]):
        name_to_idx[name] = idx
        cond_x.append(node_sizes[name])
        is_macros_list.append(node_macros.get(name, False))

    cond_x = torch.tensor(cond_x)
    is_macros = torch.tensor(is_macros_list)
    is_ports = torch.zeros_like(is_macros)

    # Build edges
    edge_indices = []
    edge_attrs = []
    edge_pin_ids = []
    for net in nets:
        src = net["output"]
        if src.obj_name not in name_to_idx:
            continue
        for sink in net["inputs"]:
            if sink.obj_name not in name_to_idx:
                continue
            edge_indices.append((name_to_idx[src.obj_name], name_to_idx[sink.obj_name]))
            edge_attrs.append((*src.offset, *sink.offset))
            edge_pin_ids.append((src.id, sink.id))

    edge_index_uni = torch.tensor(edge_indices, dtype=torch.int64)
    edge_attr_uni = torch.tensor(edge_attrs)
    edge_pin_id_uni = torch.tensor(edge_pin_ids, dtype=torch.int64)

    # Adjust pin offsets to be relative to cell center (not corner)
    u_shape = cond_x[edge_index_uni[:, 0]]
    v_shape = cond_x[edge_index_uni[:, 1]]
    edge_attr_uni[:, :2] = edge_attr_uni[:, :2] + u_shape / 2
    edge_attr_uni[:, 2:4] = edge_attr_uni[:, 2:4] + v_shape / 2

    # Make bidirectional
    e = edge_index_uni.shape[0]
    edge_index = flip_stack(edge_index_uni).T
    edge_attr = flip_stack(edge_attr_uni.view(e, 2, 2)).view(e * 2, 4)
    edge_pin_id = flip_stack(edge_pin_id_uni)

    cond = Data(
        x=cond_x / SCALING_UNITS,
        edge_index=edge_index,
        edge_attr=edge_attr / SCALING_UNITS,
        edge_pin_id=edge_pin_id,
        is_ports=is_ports,
        is_macros=is_macros,
        name_index_mapping=name_to_idx,
    )
    return x / SCALING_UNITS, cond


def download_ispd2005(target_dir):
    """Download ISPD2005 benchmarks if not present."""
    if os.path.exists(os.path.join(target_dir, "ispd2005", "adaptec1")):
        print("ISPD2005 already downloaded")
        return

    url = "http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz"
    print(f"Downloading ISPD2005 from {url}...")
    os.makedirs(target_dir, exist_ok=True)

    import subprocess
    tmp = "/tmp/ispd2005dp.tar.xz"
    subprocess.run(["wget", "-q", url, "-O", tmp], check=True)

    # Try xz + tar first, fall back to pyunpack
    try:
        subprocess.run(["tar", "xJf", tmp, "-C", target_dir], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        from pyunpack import Archive
        Archive(tmp).extractall(target_dir)

    print("ISPD2005 downloaded and extracted")


def main():
    parser = argparse.ArgumentParser(description="Parse ISPD2005 benchmarks")
    parser.add_argument("--benchmark-dir", type=str, default=None,
                        help="Path to extracted ISPD2005 (containing adaptec1/ etc.)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for pickle files")
    parser.add_argument("--download", action="store_true",
                        help="Download ISPD2005 if not present")
    args = parser.parse_args()

    cd_root = CHIPDIFFUSION_ROOT

    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir)
    else:
        benchmark_dir = cd_root / "benchmarks" / "ispd2005" / "ispd2005"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = cd_root / "datasets" / "graph" / "ispd2005"

    if args.download:
        download_ispd2005(str(cd_root / "benchmarks" / "ispd2005"))

    if not benchmark_dir.exists():
        print(f"Benchmark directory not found: {benchmark_dir}")
        print("Run with --download to fetch ISPD2005, or specify --benchmark-dir")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing ISPD2005 from {benchmark_dir}")
    print(f"Output to {output_dir}")
    print()

    for idx, name in enumerate(ISPD_NAMES):
        graph_path = output_dir / f"graph{idx}.pickle"
        output_path = output_dir / f"output{idx}.pickle"

        if graph_path.exists() and output_path.exists():
            print(f"  SKIP {name} (already parsed)")
            continue

        print(f"  Parsing {name}...", end=" ", flush=True)
        circuit_dir = benchmark_dir / name
        x, cond = parse_bookshelf(
            str(circuit_dir / f"{name}.nodes"),
            str(circuit_dir / f"{name}.nets"),
            str(circuit_dir / f"{name}.pl"),
        )
        cond.chip_size = ISPD_CHIP_SIZES[idx]

        with open(graph_path, "wb") as f:
            pickle.dump(cond, f)
        with open(output_path, "wb") as f:
            pickle.dump(x, f)

        print(f"OK ({cond.x.shape[0]} nodes, {cond.edge_index.shape[1]} edges)")

    # Write config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write("train_samples: 0\nval_samples: 8\nscale: 1\n")

    print(f"\nDone. {len(list(output_dir.glob('graph*.pickle')))} circuits parsed.")


if __name__ == "__main__":
    main()
