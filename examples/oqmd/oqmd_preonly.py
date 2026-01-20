import os
import pdb
import json
import torch
import torch_geometric
from torch_geometric.transforms import RadiusGraph, Distance, Spherical, LocalCartesian
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import argparse

import os, json
import pickle, csv
from pathlib import Path

import logging
import sys
from mpi4py import MPI

info = logging.info

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

def main():
    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--pretrained_model_ensemble_path", help="directory for ensemble of models", type=str, default="pretrained_model_ensemble"
    )

    parser.add_argument(
        "--finetuning_config", help="path to JSON file with configuration for fine-tunable architecture", type=str,
        default="./finetuning_config.json"
    )
    parser.add_argument("--log", help="log name")
    parser.add_argument("--datasetname", help="dataset name", default="oqmd")
    parser.add_argument("--modelname", help="model name", default="oqmd")

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

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    datasetname = "FineTuning" if args.datasetname is None else args.datasetname
    modelname = "FineTuning" if args.modelname is None else args.modelname

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_config.json")
    with open(filename, "r") as f:
        ft_config = json.load(f)

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "pos"]
    node_feature_dims = [1, 3]

    verbosity = ft_config["Verbosity"]["level"]
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    log_name = "oqmd_finetuning"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the oqmd/processed/ directory is removed.
    dataset = torch.load("examples/oqmd/oqmd_dataset.pt", weights_only=False)
    
    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        dataset, ft_config["NeuralNetwork"]["Training"]["perc_train"], False
    )

    print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

    print("Before COMM.Barrier()", flush=True)
    comm.Barrier()
    print("After COMM.Barrier()", flush=True)

    deg = gather_deg(trainset)
    ft_config["pna_deg"] = deg

    setnames = ["trainset", "valset", "testset"]

    ## adios
    if args.format == "adios":
        fname = os.path.join(
            os.path.dirname(__file__), "./dataset/%s.bp" % datasetname
        )
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

    ## pickle
    elif args.format == "pickle":
        basedir = os.path.join(
            os.path.dirname(__file__), "../../dataset", "%s.pickle" % datasetname
        )
        attrs = dict()
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
            attrs=attrs,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
    sys.exit(0)


if __name__ == "__main__":

    main()