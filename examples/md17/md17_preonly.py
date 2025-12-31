#!/usr/bin/env python3
import os
import pdb
import json
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch_geometric
from mpi4py import MPI
from torch_geometric.transforms import AddLaplacianEigenvectorPE, Distance

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None

log = logging.getLogger(__name__).info

# ----------------------------- Pre/Transform helpers -----------------------------
# Distance transform to ensure data.edge_attr exists (||x_i - x_j||, shape [E,1])
dist_transform = Distance(norm=False, cat=False)

def md17_pre_transform(data, compute_edges, pe_transform):
    # Node features: [Z, x, y, z] to mirror the QM9 schema (dims [1,3])
    data.x = torch.cat((data.z.float().view(-1, 1), data.pos), dim=1)

    # Build edges via HydraGNN's configured radius graph
    data = compute_edges(data)

    # Create edge_attr as Euclidean distances
    data = dist_transform(data)

    # Add Laplacian eigenvector PE
    data = pe_transform(data)

    # Optional: relative PE difference per edge
    if hasattr(data, "pe") and data.edge_index is not None:
        src, dst = data.edge_index[0], data.edge_index[1]
        data.rel_pe = torch.abs(data.pe[src] - data.pe[dst])

    # Target: per-atom energy (parity with QM9 preonly)
    data.y = data.energy / len(data.x)
    return data

# Probabilistic pre-filter (~25%). Delete processed/ to re-apply.
def md17_pre_filter(_):
    return torch.rand(1) < 0.25

def main():
    # FIX random seed
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--pretrained_model_ensemble_path",
        help="directory for ensemble of models",
        type=str,
        default="pretrained_model_ensemble",
    )
    parser.add_argument(
        "--finetuning_config",
        help="path to JSON file with configuration for fine-tunable architecture",
        type=str,
        default="./finetuning_config.json",
    )
    parser.add_argument("--log", help="log name")
    parser.add_argument("--modelname", help="model name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")
    args = parser.parse_args()

    # SERIALIZED_DATA_PATH
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    # Always initialize DDP (same pattern as QM9)
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD
    # OK if called twice in this codebase
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    modelname = "FineTuning" if args.modelname is None else args.modelname

    # Load finetuning config (use the file adjacent to this script if not overridden)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_config.json")
    if args.finetuning_config is not None and os.path.exists(args.finetuning_config):
        cfg_path = args.finetuning_config
    with open(cfg_path, "r") as f:
        ft_config = json.load(f)

    # Feature schema (parity with QM9)
    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates"]
    node_feature_dims = [1, 3]

    verbosity = ft_config["Verbosity"]["level"]
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    log_name = "md17_finetuning"
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # MD17 uracil filename quirk
    torch_geometric.datasets.MD17.file_names["revised uracil"] = "md17_uracil.npz"

    # Build HydraGNN edge constructor from config (robust defaults)
    arch_config = dict(ft_config.get("NeuralNetwork", {}).get("Architecture", {}))
    arch_config.setdefault("radius", 7.0)
    arch_config.setdefault("max_neighbours", 5)
    arch_config.setdefault("loop", False)
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # Laplacian PE transform (default k=8 if not present)
    pe_dim = arch_config.get("pe_dim", 8)
    pe_transform = AddLaplacianEigenvectorPE(k=pe_dim, attr_name="pe", is_undirected=True)

    # Build MD17 dataset (uracil) with transforms/filters.
    # NOTE: Delete dataset/md17/uracil/processed to force re-run if you change transforms or pre_filter.
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name="uracil",
        pre_transform=lambda d: md17_pre_transform(d, compute_edges, pe_transform),
        pre_filter=md17_pre_filter,
    )

    # Split like QM9 example (uses perc_train from JSON)
    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        dataset, ft_config["NeuralNetwork"]["Training"]["perc_train"], False
    )

    print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))
    print("Before COMM.Barrier()", flush=True)
    comm.Barrier()
    print("After COMM.Barrier()", flush=True)

    # Degree histogram (for PNA; harmless otherwise)
    deg = gather_deg(trainset)
    ft_config["pna_deg"] = deg

    if args.format == "adios":
        if AdiosWriter is None:
            raise ImportError("ADIOS support not available. Install hydragnn adios deps.")
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

    elif args.format == "pickle":
        basedir = os.path.join(os.path.dirname(__file__), "../../dataset", "%s.pickle" % modelname)
        attrs = {"pna_deg": deg}
        SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True, attrs=attrs)
        SimplePickleWriter(valset,   basedir, "valset",   use_subdir=True)
        SimplePickleWriter(testset,  basedir, "testset",  use_subdir=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
