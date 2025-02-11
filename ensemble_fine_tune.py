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
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.time_utils import Timer

# from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.distributed import (
    setup_ddp,
    get_distributed_model,
    print_peak_memory,
)

from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import nsplit
import hydragnn.utils.tracer as tr

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
from ensemble_utils import model_ensemble, train_ensemble, test_ensemble

def run(argv):
    assert (
        len(argv) == 4
    ), f"Usage: {argv[0]} <ensemble_path>> <ft_config.json> <dataset.bp>"

    ensemble_path = argv[1]
    ftcfgfile = argv[2]
    dataset = argv[3]
    log_name = "experiment"
    (Path("logs") / log_name).mkdir(exist_ok=True, parents=True)
    verbosity = 1

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    with open(ftcfgfile, "r") as f:
        ft_config = json.load(f)

    world_size, world_rank = hydragnn.utils.setup_ddp()
    verbosity = 2
    model = model_ensemble([os.path.join(ensemble_path,model_id) for model_id in os.listdir(ensemble_path)], fine_tune_config=ft_config)
    model = hydragnn.utils.get_distributed_model(model, verbosity)
    use_torch_backend = False  # Fix to MPI backend
    if True:  # fix to adios format
        shmem = ddstore = False
        if use_torch_backend:
            shmem = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "torch"
            os.environ["HYDRAGNN_USE_ddstore"] = "0"
        else:
            ddstore = True
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
            os.environ["HYDRAGNN_USE_ddstore"] = "1"
        
        # opt = {"preload": False, "shmem": shmem, "ddstore": ddstore, "var_config": ft_config['Variables_of_interest'] }
        opt = {"preload": False, "shmem": shmem, "ddstore": False, "var_config": model.module.var_config}
        comm = MPI.COMM_WORLD
        trainset = AdiosDataset(dataset, "trainset", comm, **opt)
        valset = AdiosDataset(dataset, "valset", comm, **opt)
        testset = AdiosDataset(dataset, "testset", comm, **opt)
        # comm.Barrier()

    print("Loaded dataset.")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    # first hurdle - we need to get metadata (what features are present) from adios datasets.
    (
        train_loader,
        val_loader,
        test_loader
    ) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, ft_config["Training"]["batch_size"]
    )

    print("Created Dataloaders")
    comm.Barrier()

    timer.stop()
    
    # Create optimizers for each ensemble member
    optimizers = [torch.optim.Adam(member.parameters(), lr=ft_config["Training"]["Optimizer"]["learning_rate"]) for member in model.module.model_ens]

    # Train the ensemble
    train_ensemble(model, train_loader, val_loader, num_epochs=ft_config["Training"]["num_epoch"], optimizers=optimizers, device="cuda")
    #Test the ensemble
    test_ensemble(model, test_loader, verbosity=2)

if __name__ == "__main__":
    import sys

    run(sys.argv)
