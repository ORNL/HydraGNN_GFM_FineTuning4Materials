#!/usr/bin/env python3

""" Run HydraGNN's main train/test/validate
    loop on the given dataset / model combination.
"""
import os, json
import pickle, csv
from pathlib import Path

import logging
import sys
from tqdm import tqdm

info = logging.info


import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

from itertools import chain
import argparse
import time

import hydragnn
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer

from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)

# from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.distributed import (
    setup_ddp,
    get_distributed_model,
    print_peak_memory,
)

from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.utils.distributed import nsplit
import hydragnn.utils.profiling_and_tracing.tracer as tr

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

# from debug_dict import DebugDict
from update_model import update_model, update_ensemble
from ensemble_utils import update_config_ensemble, model_ensemble, train_ensemble, test_ensemble

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)

    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--pretrained_model_ensemble_path", help="directory for ensemble of models", type=str, default="pretrained_model_ensemble"
    )

    parser.add_argument(
        "--finetuning_config", help="path to JSON file with configuration for fine-tunable architecture", type=str,
        default="./finetuning_config.json"
    )
    parser.add_argument("--log", help="log name")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
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
    ensemble_path = Path("pretrained_model_ensemble")

    log_name = "experiment"
    (Path("logs") / log_name).mkdir(exist_ok=True, parents=True)
    verbosity = 1

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    ftcfgfile = args.finetuning_config

    with open(ftcfgfile, "r") as f:
        ft_config = json.load(f)

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

    if args.batch_size is not None:
        ft_config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "MPTrj" if args.log is None else args.log
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "FineTuning" if args.modelname is None else args.modelname

    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    verbosity = 2

    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)

    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    print("Loaded dataset.")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    # first hurdle - we need to get metadata (what features are present) from adios datasets.
    (
        train_loader,
        val_loader,
        test_loader
    ) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, ft_config["NeuralNetwork"]["Training"]["batch_size"]
    )

    ft_config = hydragnn.utils.input_config_parsing.update_config(
        ft_config, train_loader, val_loader, test_loader
    )

    model_dir_list = [os.path.join(ensemble_path, model_id) for model_id in os.listdir(ensemble_path)]
    update_config_ensemble(model_dir_list, train_loader, val_loader, test_loader)

    model = model_ensemble(model_dir_list, fine_tune_config=ft_config, GFM_2024=True)
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
    use_torch_backend = False  # Fix to MPI backend

    print("Created Dataloaders")
    comm.Barrier()

    timer.stop()
    
    # Create optimizers for each ensemble member
    optimizers = [torch.optim.Adam(member.parameters(), lr=ft_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]) for member in model.module.model_ens]

    # Train the ensemble
    train_ensemble(model, train_loader, val_loader, num_epochs=ft_config["NeuralNetwork"]["Training"]["num_epoch"], optimizers=optimizers, device="cuda")
    #Test the ensemble
    test_ensemble(model, test_loader, 'qm9',  verbosity=2)
