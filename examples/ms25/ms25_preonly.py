#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import torch
from torch_geometric.transforms import Distance
from torch_geometric.data import Data

from mpi4py import MPI
import hydragnn
from hydragnn.utils.datasets.pickledataset import SimplePickleWriter
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

from ase.io.trajectory import Trajectory
from ase.io import read


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

HARTREE_TO_EV = 27.2114
dist_transform = Distance(norm=False, cat=False)


def pick_traj_files(system_dir: Path):
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
    if atoms.calc is not None:
        return float(atoms.get_potential_energy()), "calc"
    if "E" in atoms.info:
        return float(atoms.info["E"]) * HARTREE_TO_EV, "info[E]_converted"
    raise RuntimeError("No energy found: no calc and no info['E'].")


def compute_chemical_composition(z: torch.Tensor) -> torch.Tensor:
    comp = torch.zeros(118, dtype=torch.float64)
    z_flat = z.view(-1).to(torch.long)
    for val in z_flat.tolist():
        if 1 <= val <= 118:
            comp[val - 1] += 1.0
    return comp


def ase_atoms_to_pyg_data(atoms, dataset_name="ms25"):
    """
    Convention for merged codebase:
    - data.energy        = total energy in eV        (for MLIP loss)
    - data.y             = per-atom energy (eV/atom)  (regression target)
    - data.energy_per_atom = eV/atom

    With energy_target_mode: "total" in config, _get_training_targets
    multiplies data.y * natoms to get total energy for the loss,
    and reports per-atom MAE via _convert_predictions_to_per_atom.
    """
    z = torch.as_tensor(atoms.numbers, dtype=torch.long)
    natoms_int = len(z)
    natoms = torch.tensor([natoms_int], dtype=torch.long)

    pos = torch.as_tensor(atoms.positions, dtype=torch.float64)
    cell = torch.as_tensor(atoms.cell.array, dtype=torch.float64).view(3, 3)
    pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool).view(3)

    x = z.view(-1, 1).to(torch.float64)
    atomic_numbers = z.view(-1, 1).to(torch.long)

    total_energy_ev, src = get_energy_from_atoms(atoms)
    per_atom_energy_ev = total_energy_ev / float(natoms_int)

    energy = torch.tensor([total_energy_ev], dtype=torch.float64)
    y = torch.tensor([[per_atom_energy_ev]], dtype=torch.float64)
    energy_per_atom = torch.tensor([per_atom_energy_ev], dtype=torch.float64)

    graph_attr = torch.tensor([0.0, 1.0], dtype=torch.float64)
    chemical_composition = compute_chemical_composition(z)

    forces = None
    try:
        if atoms.calc is not None:
            forces = torch.as_tensor(atoms.get_forces(), dtype=torch.float64)
    except Exception:
        forces = None

    data = Data(
        dataset_name=dataset_name,
        natoms=natoms,
        pos=pos,
        cell=cell,
        pbc=pbc,
        edge_index=None,
        edge_attr=None,
        atomic_numbers=atomic_numbers,
        chemical_composition=chemical_composition,
        smiles_string=None,
        x=x,
        energy=energy,
        energy_per_atom=energy_per_atom,
        forces=forces,
        graph_attr=graph_attr,
        y=y,
    )

    if forces is not None:
        data.force = forces
    data.energy_source = src
    return data


def build_edges_and_features(data, compute_edges):
    data = compute_edges(data)
    data = dist_transform(data)
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(torch.float64)
    return data


def iter_frames(traj_path: Path, max_frames: int, stride: int):
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
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--ms25_root", type=str, default="/home/l54/Datasets")
    parser.add_argument("--out_root", type=str, default=str((here / "../../dataset").resolve()))
    parser.add_argument("--finetuning_config", type=str, default=str((here / "finetuning_config.json").resolve()))
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--pickle_tag", type=str, default="mlip_peratom")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD
    world_size, rank = hydragnn.utils.distributed.get_comm_size_and_rank()

    ms25_root = Path(args.ms25_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.finetuning_config, "r") as f:
        base_cfg = json.load(f)

    all_dirs = sorted([p.name for p in ms25_root.iterdir() if p.is_dir()])
    systems = [args.system] if args.system is not None else [s for s in all_dirs if s in MS25_CUTOFFS]

    if rank == 0:
        print(f"[ROOT] {ms25_root}")
        print(f"[OUT ] {out_root}")
        print(f"[SYSTEMS] {systems}")
        print(f"[TAG ] {args.pickle_tag}")

    for sysname in systems:
        if sysname not in MS25_CUTOFFS:
            if rank == 0:
                print(f"[SKIP] {sysname}")
            continue

        radius, max_nbrs = MS25_CUTOFFS[sysname]
        sysdir = ms25_root / sysname

        cfg = json.loads(json.dumps(base_cfg))
        arch = cfg["NeuralNetwork"]["Architecture"]
        arch["radius"] = float(radius)
        arch["max_neighbours"] = int(max_nbrs)
        arch["periodic_boundary_conditions"] = True

        var = cfg["NeuralNetwork"]["Variables_of_interest"]
        var["graph_feature_names"] = ["energy"]
        var["graph_feature_dims"] = [1]
        var["node_feature_names"] = ["atomic_number", "cartesian_coordinates"]
        var["node_feature_dims"] = [1, 3]
        var["input_node_features"] = [0]

        compute_edges = hydragnn.preprocess.get_radius_graph_config(arch)
        traj_files = pick_traj_files(sysdir)

        if rank == 0:
            print(f"\n[{sysname}] radius={radius} max_neighbours={max_nbrs}")
            print(f"[{sysname}] trajs: {[str(p.relative_to(ms25_root)) for p in traj_files]}")

        data_list = []
        energy_src_counts = {}
        for tp in traj_files:
            for atoms in iter_frames(tp, max_frames=args.max_frames, stride=args.stride):
                d = ase_atoms_to_pyg_data(atoms, dataset_name="ms25")
                d = build_edges_and_features(d, compute_edges)
                src = str(getattr(d, "energy_source", "unknown"))
                energy_src_counts[src] = energy_src_counts.get(src, 0) + 1
                data_list.append(d)

        if rank == 0:
            print(f"[{sysname}] frames loaded: {len(data_list)}  energy_src={energy_src_counts}")

        trainset, valset, testset = hydragnn.preprocess.split_dataset(
            data_list, cfg["NeuralNetwork"]["Training"]["perc_train"], False,
        )
        comm.Barrier()

        deg = gather_deg(trainset)
        attrs = {
            "pna_deg": deg, "radius": radius, "max_neighbours": max_nbrs,
            "target_mode": "per_atom_energy", "energy_units": "eV/atom",
            "stride": args.stride,
            "num_train": len(trainset), "num_val": len(valset), "num_test": len(testset),
        }

        if rank == 0 and len(trainset) > 0:
            s = trainset[0]
            print(f"[{sysname}] [SANITY] y (eV/atom):", s.y.view(-1)[0].item())
            print(f"[{sysname}] [SANITY] energy (eV):", s.energy.view(-1)[0].item())
            print(f"[{sysname}] [SANITY] natoms:", s.natoms.item())
            print(f"[{sysname}] [SANITY] y*natoms (=energy?):", s.y.view(-1)[0].item() * s.natoms.item())

        pickle_name = f"{sysname}_{args.pickle_tag}.pickle"
        basedir = out_root / pickle_name
        if rank == 0:
            print(f"[{sysname}] writing -> {basedir}")

        SimplePickleWriter(trainset, str(basedir), "trainset", use_subdir=True, attrs=attrs)
        SimplePickleWriter(valset, str(basedir), "valset", use_subdir=True)
        SimplePickleWriter(testset, str(basedir), "testset", use_subdir=True)
        comm.Barrier()

    if rank == 0:
        print("\n[DONE] MS25 preprocessing complete.")


if __name__ == "__main__":
    main()