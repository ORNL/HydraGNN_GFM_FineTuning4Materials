#!/usr/bin/env python3
import os
import sys
import json
import torch
import argparse
import torch_geometric
from torch_geometric.transforms import Distance
from mpi4py import MPI

import hydragnn
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.utils.datasets.pickledataset import SimplePickleWriter

try:
    from hydragnn.utils.adiosdataset import AdiosWriter
except ImportError:
    AdiosWriter = None


transform_coordinates = Distance(norm=False, cat=False)

num_samples = 1000

def qm9_pre_filter(data):
    return data.idx < num_samples


def qm9_pre_transform(data):
    """
    Convention for merged codebase:
    - data.y             = per-atom atomization energy (eV/atom) — regression target
    - data.energy        = total atomization energy (eV)
    - data.energy_per_atom = eV/atom (same as data.y)

    With energy_target_mode: "total" in config, _get_training_targets
    multiplies data.y * natoms → total energy for loss,
    and reports per-atom MAE via _convert_predictions_to_per_atom.
    """
    z = data.z.view(-1).long()
    natoms = z.numel()

    x = z.view(-1, 1).to(torch.float64)
    pos = data.pos.to(torch.float64)

    data.x = x
    data.pos = pos
    data = transform_coordinates(data)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(torch.float64)

    total_energy    = data.y[:, 12].view(1, 1).to(torch.float64)
    energy_per_atom = total_energy / float(natoms)

    data.x              = x
    data.y              = energy_per_atom        # per-atom target
    data.energy         = total_energy
    data.energy_per_atom = energy_per_atom
    data.natoms         = torch.tensor([natoms], dtype=torch.long)
    data.graph_attr     = torch.tensor([0.0, 1.0], dtype=torch.float64)

    return data


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--shmem", action="store_true")
    parser.add_argument("--finetuning_config", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_config.json"))
    parser.add_argument("--log",         type=str, default="qm9_preonly")
    parser.add_argument("--datasetname", type=str, default="qm9_fast_preonly")
    parser.add_argument("--modelname",   type=str, default="qm9_fast_preonly")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adios",  action="store_const", dest="format", const="adios")
    group.add_argument("--pickle", action="store_const", dest="format", const="pickle")
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    if "SERIALIZED_DATA_PATH" not in os.environ:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    with open(args.finetuning_config, "r") as f:
        ft_config = json.load(f)

    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = ["energy"]
    var_config["graph_feature_dims"]  = [1]
    var_config["node_feature_names"]  = ["atomic_number", "cartesian_coordinates"]
    var_config["node_feature_dims"]   = [1, 3]

    hydragnn.utils.print.print_utils.setup_log(args.log)

    if rank == 0:
        print("[INFO] Target: y = per-atom atomization energy (eV/atom)", flush=True)
        print("[INFO] Loading QM9 (delete dataset/qm9/processed/ if rerunning)", flush=True)

    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9",
        pre_transform=qm9_pre_transform,
        pre_filter=qm9_pre_filter,
    )

    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        dataset, ft_config["NeuralNetwork"]["Training"]["perc_train"], False,
    )

    if rank == 0:
        print(f"[INFO] dataset={len(dataset)}  train={len(trainset)} val={len(valset)} test={len(testset)}", flush=True)
        if len(trainset) > 0:
            s = trainset[0]
            print(f"[SANITY] y (eV/atom):       {s.y.view(-1)[0].item():.6f}", flush=True)
            print(f"[SANITY] energy (eV):       {s.energy.view(-1)[0].item():.6f}", flush=True)
            print(f"[SANITY] natoms:            {s.natoms.item()}", flush=True)
            print(f"[SANITY] y*natoms = energy? {s.y.view(-1)[0].item()*s.natoms.item():.6f}", flush=True)

    comm.Barrier()

    deg = gather_deg(trainset)
    ft_config["pna_deg"] = deg

    if args.format == "adios":
        if AdiosWriter is None:
            raise ImportError("AdiosWriter not available.")
        fname = os.path.join(os.path.dirname(__file__), f"./dataset/{args.datasetname}.bp")
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset",   valset)
        adwriter.add("testset",  testset)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()
        if rank == 0:
            print(f"[INFO] Saved ADIOS to {fname}", flush=True)

    elif args.format == "pickle":
        basedir = os.path.join(os.path.dirname(__file__), "../../dataset", f"{args.datasetname}.pickle")
        if rank == 0:
            print(f"[INFO] Writing pickle to {basedir}", flush=True)
        attrs = {"pna_deg": deg}
        SimplePickleWriter(trainset, basedir, "trainset", use_subdir=True, attrs=attrs)
        SimplePickleWriter(valset,   basedir, "valset",   use_subdir=True)
        SimplePickleWriter(testset,  basedir, "testset",  use_subdir=True)
        if rank == 0:
            print(f"[INFO] Done.", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()