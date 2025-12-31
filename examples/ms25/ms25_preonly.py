#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch_geometric.transforms import Distance

from mpi4py import MPI
import hydragnn
from hydragnn.utils.datasets.pickledataset import SimplePickleWriter
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

from ase.io.trajectory import Trajectory 
from ase.io import read  


# (radius [Å], max_neighbours)
MS25_CUTOFFS = {
    "MgO-2x2":  (6.0, 64),
    "MgO-4x4":  (6.0, 64),
    "H2O-64":   (6.5, 48),
    "H2O-192":  (6.5, 48),
    "CHA":      (5.0, 64),
    "HEA":      (5.5, 64),
    "Reaction": (6.0, 64),
    "Zr-O":     (6.0, 64),
}

dist_transform = Distance(norm=False, cat=False)


def pick_traj_files(system_dir: Path):
    """
    Prefer set1/train-1000.traj (matches your scans), else train.traj, else any .traj under set1/, else any .traj.
    """
    preferred = []
    for name in ["train-1000.traj", "train.traj", "val.traj", "test.traj", "neb.traj"]:
        p = system_dir / "set1" / name
        if p.exists():
            preferred.append(p)
    if preferred:
        return preferred

    set1 = system_dir / "set1"
    if set1.exists():
        set1_trajs = sorted(set1.rglob("*.traj"))
        if set1_trajs:
            return set1_trajs

    return sorted(system_dir.rglob("*.traj"))


def get_energy_from_atoms(atoms):
    if "E" in atoms.info:
        return float(atoms.info["E"]), "info[E]"
    if atoms.calc is not None:
        return float(atoms.get_potential_energy()), "calc"
    raise RuntimeError("No energy found: missing atoms.info['E'] and no attached calculator.")


def ase_atoms_to_pyg_data(atoms, y_mode="per_atom"):
    """
    ASE Atoms -> PyG Data with:
      z [N] long
      pos [N,3] float
      cell [3,3] float
      pbc [3] bool
      y [1,1] float (graph target)
    """
    from torch_geometric.data import Data

    z = torch.as_tensor(atoms.numbers, dtype=torch.long)
    pos = torch.as_tensor(atoms.positions, dtype=torch.float32)
    cell = torch.as_tensor(atoms.cell.array, dtype=torch.float32).view(3, 3)
    pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool).view(3)

    E, src = get_energy_from_atoms(atoms)
    if y_mode == "per_atom":
        yval = E / float(len(z))
    else:
        yval = E

    y = torch.tensor([[yval]], dtype=torch.float32)

    data = Data(z=z, pos=pos, cell=cell, pbc=pbc, y=y)
    data.energy_source = src  # for debug; non-tensor is ok here
    return data


def build_edges_and_features(data, compute_edges):
    data = compute_edges(data)
    data = dist_transform(data)  # edge_attr = distance, shape [E,1]
    return data


def standardize_splits(trainset, valset, testset):
    ys = torch.cat([d.y.view(-1) for d in trainset], dim=0)
    mean = float(ys.mean().item())
    std = float(ys.std().item())
    if std == 0.0:
        raise ValueError("Train label std is zero; cannot standardize.")
    for split in (trainset, valset, testset):
        for d in split:
            d.y = (d.y - mean) / std
    return mean, std


def iter_frames(traj_path: Path, max_frames: int, stride: int):
    """
    Stream frames from .traj. Falls back to ase.io.read if needed.
    """
    try:
        traj = Trajectory(str(traj_path))
        count = 0
        for i, atoms in enumerate(traj):
            if stride > 1 and (i % stride != 0):
                continue
            yield atoms
            count += 1
            if max_frames > 0 and count >= max_frames:
                break
    except Exception:
        frames = read(str(traj_path), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        if stride > 1:
            frames = frames[::stride]
        if max_frames > 0:
            frames = frames[:max_frames]
        for atoms in frames:
            yield atoms


def main():
    here = Path(__file__).resolve().parent  # examples/ms25

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms25_root", type=str, default="/Downloads/Datasets",
        help="Directory containing MS25 system folders (CHA, HEA, ...)."
    )
    parser.add_argument(
        "--out_root", type=str, default=str((here / "dataset").resolve()),
        help="Write dataset/<SYSTEM>.pickle/... under this directory."
    )
    parser.add_argument(
        "--finetuning_config", type=str, default=str((here / "finetuning_config.json").resolve()),
        help="Base JSON config to load + patch Variables_of_interest + Architecture cutoffs."
    )
    parser.add_argument("--system", type=str, default=None,
                        help="If set, preprocess only this system (e.g., HEA). Otherwise do all systems found.")
    parser.add_argument("--y_mode", type=str, choices=["per_atom", "total"], default="per_atom",
                        help="Store target as per-atom energy or total energy.")
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize y using train mean/std (saved as attrs).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="If >0, cap number of frames per traj (debug).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every k-th frame (debug/downsample).")
    args = parser.parse_args()

    # DDP init (matches your example style)
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    ms25_root = Path(args.ms25_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.finetuning_config, "r") as f:
        base_cfg = json.load(f)

    all_dirs = sorted([p.name for p in ms25_root.iterdir() if p.is_dir()])
    if args.system is not None:
        systems = [args.system]
    else:
        systems = [s for s in all_dirs if s in MS25_CUTOFFS]

    if rank == 0:
        print(f"[ROOT] {ms25_root}")
        print(f"[OUT ] {out_root}")
        print(f"[SYSTEMS] {systems}")

    for sysname in systems:
        if sysname not in MS25_CUTOFFS:
            if rank == 0:
                print(f"[SKIP] {sysname}: not in MS25_CUTOFFS")
            continue

        radius, max_nbrs = MS25_CUTOFFS[sysname]
        sysdir = ms25_root / sysname

        cfg = json.loads(json.dumps(base_cfg))  # deep copy via json
        arch = cfg["NeuralNetwork"]["Architecture"]
        arch["radius"] = float(radius)
        arch["max_neighbours"] = int(max_nbrs)
        arch["periodic_boundary_conditions"] = True

        # Feature schema 
        var = cfg["NeuralNetwork"]["Variables_of_interest"]
        var["graph_feature_names"] = ["energy"]
        var["graph_feature_dims"] = [1]
        var["node_feature_names"] = ["atomic_number", "cartesian_coordinates"]
        var["node_feature_dims"] = [1, 3]

        compute_edges = hydragnn.preprocess.get_radius_graph_config(arch)

        traj_files = pick_traj_files(sysdir)
        if rank == 0:
            rel = [str(p.relative_to(ms25_root)) for p in traj_files]
            print(f"\n[{sysname}] radius={radius} max_neighbours={max_nbrs}")
            print(f"[{sysname}] trajs: {rel}")

        data_list = []
        energy_src_counts = {"info[E]": 0, "calc": 0}

        for tp in traj_files:
            for atoms in iter_frames(tp, max_frames=args.max_frames, stride=args.stride):
                d = ase_atoms_to_pyg_data(atoms, y_mode=args.y_mode)
                # x = [Z, x, y, z]
                d.x = torch.cat((d.z.float().view(-1, 1), d.pos), dim=1)
                d = build_edges_and_features(d, compute_edges)

                src = str(getattr(d, "energy_source", "unknown"))
                energy_src_counts[src] = energy_src_counts.get(src, 0) + 1
                data_list.append(d)

        if rank == 0:
            print(f"[{sysname}] frames loaded: {len(data_list)}  energy_src={energy_src_counts}")

        trainset, valset, testset = hydragnn.preprocess.split_dataset(
            data_list, cfg["NeuralNetwork"]["Training"]["perc_train"], False
        )

        comm.Barrier()

        deg = gather_deg(trainset)
        attrs = {
            "pna_deg": deg,
            "radius": radius,
            "max_neighbours": max_nbrs,
            "y_mode": args.y_mode,
            "stride": args.stride,
        }

        if args.standardize:
            mean, std = standardize_splits(trainset, valset, testset)
            attrs["label_mean"] = mean
            attrs["label_std"] = std
            if rank == 0:
                print(f"[{sysname}] standardized: mean={mean:.6f}, std={std:.6f}")

        basedir = out_root / f"{sysname}.pickle"
        if rank == 0:
            print(f"[{sysname}] writing -> {basedir}")

        SimplePickleWriter(trainset, str(basedir), "trainset", use_subdir=True, attrs=attrs)
        SimplePickleWriter(valset,   str(basedir), "valset",   use_subdir=True)
        SimplePickleWriter(testset,  str(basedir), "testset",  use_subdir=True)

        comm.Barrier()

    if rank == 0:
        print("\n[DONE] MS25 preprocessing complete.")


if __name__ == "__main__":
    main()

