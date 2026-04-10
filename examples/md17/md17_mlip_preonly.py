#!/usr/bin/env python3
"""
MD17 preprocessing for MLIP fine-tuning (energy + forces).

Creates data with:
  - data.x = atomic_number   (shape [N, 1])
  - data.pos = positions      (shape [N, 3])
  - data.energy = total energy (scalar)
  - data.forces = force vectors (shape [N, 3])
  - data.y = per-atom energy   (shape [N, 1])  -- target for graph-level head
  - data.graph_attr = [charge, spin] = [0.0, 1.0]
"""
import os
import sys
import json
import argparse
import logging

import torch
import torch_geometric
from torch_geometric.transforms import Distance
from mpi4py import MPI

import hydragnn
from hydragnn.utils.datasets.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.adiosdataset import AdiosWriter
except ImportError:
    AdiosWriter = None

log = logging.getLogger(__name__).info
dist_transform = Distance(norm=False, cat=False)

# ---- constants ----
KCAL_PER_EV = 23.0609  # 1 eV = 23.0609 kcal/mol

# ---- configurable cap ----
MAX_SAMPLES = 1000


def md17_mlip_pre_transform(data, compute_edges):
    """Transform a raw MD17 sample for MLIP training."""

    # Node features: just atomic number (shape [N, 1])
    data.x = data.z.float().view(-1, 1)

    # Positions are already in data.pos (shape [N, 3])

    # Energy: total energy (scalar) — will be shifted to relative after loading
    data.energy = data.energy.squeeze()

    # Forces: per-atom force vectors (shape [N, 3])
    # Forces are invariant to a constant energy shift (F = -dE/dr)
    data.forces = data.force.float()

    # Placeholder y — will be recomputed after energy shift
    data.y = data.energy / len(data.x)

    # Graph attributes for conditioning (charge=0, spin=1)
    data.graph_attr = torch.tensor([0.0, 1.0], dtype=torch.float32)

    # Build edges
    data = compute_edges(data)
    data = dist_transform(data)

    return data


_pre_filter_counter = 0


def md17_mlip_pre_filter(data):
    global _pre_filter_counter
    if _pre_filter_counter >= MAX_SAMPLES:
        return False
    _pre_filter_counter += 1
    return True


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--shmem", action="store_true")
    parser.add_argument(
        "--finetuning_config",
        type=str,
        default=None,
        help="path to JSON config",
    )
    parser.add_argument("--datasetname", default="md17_mlip")
    parser.add_argument("--molecule", default="uracil",
                        help="MD17 molecule name (e.g. uracil, aspirin, ethanol)")
    parser.add_argument("--max_samples", type=int, default=1000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios", action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="pickle")
    args = parser.parse_args()

    global MAX_SAMPLES
    MAX_SAMPLES = args.max_samples

    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    # Load config
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finetuning_config_mlip.json"
    )
    if args.finetuning_config and os.path.exists(args.finetuning_config):
        cfg_path = args.finetuning_config
    with open(cfg_path, "r") as f:
        ft_config = json.load(f)

    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = ["energy"]
    var_config["graph_feature_dims"] = [1]
    var_config["node_feature_names"] = ["atomic_number"]
    var_config["node_feature_dims"] = [1]

    hydragnn.utils.print.print_utils.setup_log("md17_mlip_preprocess")

    # Fix MD17 filename quirks
    torch_geometric.datasets.MD17.file_names["revised uracil"] = "md17_uracil.npz"

    # Edge builder from config
    arch_config = dict(ft_config.get("NeuralNetwork", {}).get("Architecture", {}))
    arch_config.setdefault("radius", 5.0)
    arch_config.setdefault("max_neighbours", 20)
    arch_config.setdefault("loop", False)
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # Build dataset
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name=args.molecule,
        pre_transform=lambda d: md17_mlip_pre_transform(d, compute_edges),
        pre_filter=md17_mlip_pre_filter,
    )

    print(f"Loaded {len(dataset)} samples for {args.molecule}")
    print(f"  Sample 0: x.shape={dataset[0].x.shape}, pos.shape={dataset[0].pos.shape}")
    print(f"            energy (raw)={dataset[0].energy.item():.4f}")
    print(f"            forces.shape={dataset[0].forces.shape}")

    # --- Shift energies to relative (subtract mean) ---
    # Raw MD17 energies are absolute (~-260120 kcal/mol for uracil).
    # Forces are invariant to a constant energy shift (F = -dE/dr).
    # Materialize into a list first — PyG datasets read from disk cache,
    # so in-place modifications on dataset[i] don't persist.
    data_list = [dataset[i].clone() for i in range(len(dataset))]

    all_energies = torch.tensor([d.energy.item() for d in data_list])
    mean_energy = all_energies.mean()
    print(f"  Energy stats (raw): mean={mean_energy.item():.4f}, "
          f"std={all_energies.std().item():.4f}, "
          f"min={all_energies.min().item():.4f}, max={all_energies.max().item():.4f}")

    for d in data_list:
        d.energy = d.energy - mean_energy
        num_atoms = d.x.shape[0]
        d.y = (d.energy / num_atoms).unsqueeze(0)

    shifted_energies = torch.tensor([d.energy.item() for d in data_list])
    print(f"  Energy stats (shifted, kcal/mol): mean={shifted_energies.mean().item():.4f}, "
          f"std={shifted_energies.std().item():.4f}, "
          f"min={shifted_energies.min().item():.4f}, max={shifted_energies.max().item():.4f}")

    # --- Convert kcal/mol → eV (energy) and kcal/(mol·Å) → eV/Å (forces) ---
    # The pretrained GFM uses eV / eV·Å⁻¹, so we must match those units.
    for d in data_list:
        d.energy = d.energy / KCAL_PER_EV
        d.forces = d.forces / KCAL_PER_EV
        num_atoms = d.x.shape[0]
        d.y = (d.energy / num_atoms).unsqueeze(0)

    ev_energies = torch.tensor([d.energy.item() for d in data_list])
    print(f"  Energy stats (eV): mean={ev_energies.mean().item():.4f}, "
          f"std={ev_energies.std().item():.4f}, "
          f"min={ev_energies.min().item():.4f}, max={ev_energies.max().item():.4f}")

    # Split (use the materialized list, not the PyG dataset)
    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        data_list, ft_config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    print(rank, "Split:", len(trainset), len(valset), len(testset))

    comm.Barrier()

    deg = gather_deg(trainset)
    ft_config["pna_deg"] = deg

    datasetname = args.datasetname
    if args.format == "pickle":
        basedir = os.path.join(
            os.path.dirname(__file__), "../../dataset", f"{datasetname}.pickle"
        )
        attrs = {"pna_deg": deg}
        SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True, attrs=attrs)
        SimplePickleWriter(valset, basedir, "valset", use_subdir=True)
        SimplePickleWriter(testset, basedir, "testset", use_subdir=True)
        print(f"Saved to {basedir}")
    elif args.format == "adios":
        if AdiosWriter is None:
            raise ImportError("ADIOS not available")
        fname = os.path.join(os.path.dirname(__file__), f"./dataset/{datasetname}.bp")
        adw = AdiosWriter(fname, comm)
        adw.add("trainset", trainset)
        adw.add("valset", valset)
        adw.add("testset", testset)
        adw.add_global("pna_deg", deg)
        adw.save()

    sys.exit(0)


if __name__ == "__main__":
    main()
