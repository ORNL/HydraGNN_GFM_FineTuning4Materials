#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import os, sys, json, yaml, logging, random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch_geometric.data

sys.path.append('/Users/varrunprakash/Desktop/Finetuing/GFM_FineTune/data_utils')

from data_utils.models import DataDescriptor, number_categories
from data_utils.import_data import load_columns, calc_offsets
from data_utils.yaml_to_config import group_features, get_arc_config, get_training_config
from ensemble_utils import model_ensemble
from ensemble_utils import train_ensemble, test_ensemble

import mpi4py
mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

from hydragnn.utils.pickledataset import SimplePickleWriter
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    get_edge_attribute_name,
    generate_graphdata_from_smilestr,
)

from hydragnn.utils.smiles_utils import get_node_attribute_name, get_edge_attribute_name
from hydragnn.preprocess.utils import gather_deg
from hydragnn.preprocess import create_dataloaders
from hydragnn.utils import nsplit
import sys


import hydragnn.utils.tracer as tr
from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.utils.time_utils import Timer
from update_model import update_model, update_ensemble
from ensemble_utils import model_ensemble, train_ensemble, test_ensemble
import torch.distributed as dist
from hydragnn.utils.adiosdataset import AdiosWriter
from hydragnn.utils.adiosdataset import AdiosDataset

logging.getLogger("hydragnn").setLevel(logging.WARNING)
logging.getLogger("adios2").setLevel(logging.WARNING)


def generate_ft_config(yaml_file, output_config_file):
  
    with open(yaml_file, "r", encoding="utf-8") as f:
        descr = yaml.safe_load(f)
    arc_config = get_arc_config(descr)
    train_config = get_training_config(descr)
    ft_config = {
        "FTNeuralNetwork": {"Architecture": arc_config},
        "Training": train_config,
    }
    with open(output_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(ft_config, indent=2))
    print(f"Generated fine-tuning config: {output_config_file}")


def process_dataset(dataset_csv, dataset_yaml, output_folder):

    dataset_name = Path(dataset_csv).stem
    print(f"Processing dataset: {dataset_name}")

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    contents = list(out_dir.iterdir())
    if contents:
        print(
            f"Processed folder {out_dir} for {dataset_name} is not empty. "
            f"Skipping processing. Folder contents: {[c.name for c in contents]}"
        )
        return

    comm_size, rank = setup_ddp()
    comm = MPI.COMM_WORLD

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        descr = DataDescriptor.model_validate(yaml.safe_load(f))

    smiles_sets, values_sets, task_names = load_columns(dataset_csv, descr)
    print(f"trainset,valset,testset sizes: {list(map(len, smiles_sets))}")
  
    setnames = ["trainset", "valset", "testset"]
    dataset_data = {k: [] for k in setnames}
    missed_total = 0
    for name, smileset, valueset in zip(setnames, smiles_sets, values_sets):
        rx = list(nsplit(range(len(smileset)), comm_size))[rank]
        _smileset = smileset[rx.start : rx.stop]
        _valueset = valueset[rx.start : rx.stop]
        missed_count = 0
        for smilestr, ytarget in tqdm(zip(_smileset, _valueset),
            total=len(_smileset),
            disable=(rank != 0),
            desc=f"Featurizing {name}",
            leave=True,
            dynamic_ncols=True,
            ncols=100):
            try:
                data = generate_graphdata_from_smilestr(
                    smilestr, ytarget, get_positions=True, pretrained=True
                )
            
                data.edge_attr = torch.Tensor([1]).repeat(data.edge_index.shape[1]).unsqueeze(1)
                assert isinstance(data, torch_geometric.data.Data)
                dataset_data[name].append(data)
            except Exception as e:
                print(f"Exception processing molecule {smilestr}: {e}. Skipping molecule.")
                missed_count += 1
        if rank == 0:
            print(f"Finished processing {name} for dataset {dataset_name}. Missed {missed_count} molecules.")
        missed_total += missed_count
    if rank == 0:
        print(f"Total missed molecules for dataset {dataset_name}: {missed_total}")
    
    pna_deg = gather_deg(dataset_data["trainset"])

    node_names, node_dims = get_node_attribute_name()
    node_names = [x.replace("atomicnumber", "atomic_number") for x in node_names]
    edge_names, edge_dims = get_edge_attribute_name()
    task_dims = [1] * len(task_names)

    attrs = {
        "x_name": node_names,
        "x_name/feature_count": np.array(node_dims),
        "x_name/feature_offset": np.array(calc_offsets(node_dims)),
        "y_name": task_names,
        "y_name/feature_count": np.array(task_dims),
        "y_name/feature_offset": np.array(calc_offsets(task_dims)),
        "edge_attr_name": edge_names,
        "edge_attr_name/feature_count": np.array(edge_dims),
        "edge_attr_name/feature_offset": np.array(calc_offsets(edge_dims)),
    }

    adwriter = AdiosWriter(str(out_dir), comm)
    for name, data_list in dataset_data.items():
        adwriter.add(name, data_list)
    for k, v in attrs.items():
        adwriter.add_global(k, v)
    adwriter.add_global("pna_deg", pna_deg)
    adwriter.save()


def training_ensemble(processed_dataset_dir, dataset_name, ensemble_path, ft_config_file, model_save_dir):

    with open(ft_config_file, "r") as f:
        ft_config = json.load(f)

    Path("logs/experiment").mkdir(exist_ok=True, parents=True)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    world_size, world_rank = setup_ddp()
    comm = MPI.COMM_WORLD
   
    dataset_file = str(Path(processed_dataset_dir))

    model_files = [os.path.join(ensemble_path, model_id) for model_id in os.listdir(ensemble_path)]
    
    model = model_ensemble(model_files, fine_tune_config=ft_config)
    model = get_distributed_model(model, verbosity=2)
    
    use_torch_backend = False
    if use_torch_backend:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "torch"
        os.environ["HYDRAGNN_USE_ddstore"] = "0"
        opt = {"preload": False, "shmem": True, "ddstore": False, "var_config": model.module.var_config}
    else:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
        opt = {"preload": False, "shmem": False, "ddstore": False, "var_config": model.module.var_config}
    
    trainset = AdiosDataset(dataset_file, "trainset", comm, **opt)
    valset = AdiosDataset(dataset_file, "valset", comm, **opt)
    testset = AdiosDataset(dataset_file, "testset", comm, **opt)
    
    print("Loaded dataset.")
    print("trainset, valset, testset sizes: ", len(trainset), len(valset), len(testset))
   
    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, ft_config["Training"]["batch_size"]
    )
    print("Created Dataloaders.")
    comm.Barrier()
    
    timer.stop()
    
    optimizers = [torch.optim.Adam(member.parameters(), lr=ft_config["Training"]["Optimizer"]["learning_rate"])
                  for member in model.module.model_ens]

    for epoch in range(1):

        print(f"Training dataset {dataset_name}, epoch {epoch}...")
        train_ensemble(model, train_loader, val_loader, num_epochs=1, optimizers=optimizers, device="cuda")
        test_ensemble(model, test_loader, dataset_name, verbosity=2)
        
        save_dir = Path(model_save_dir) / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"model_epoch_{epoch}.pt"
        if world_rank == 0:
            torch.save(model.module.state_dict(), str(checkpoint_path))
            print(f"Saved model checkpoint to {checkpoint_path}")
            
            


def main():
    
    datasets_dir = "/Users/varrunprakash/Desktop/Finetuing/GFM_FineTune/datasets/tdc"
    processed_sets_dir = "/Users/varrunprakash/Desktop/Finetuing/GFM_FineTune/datasets/processed_sets"
    ensemble_path = "/Users/varrunprakash/Desktop/Finetuing/HydraGNN_Predictive_GFM_2024/Ensemble_of_models"
    model_save_dir = "/Users/varrunprakash/Desktop/Finetuing/GFM_FineTune/trained_models"

    all_files = os.listdir(datasets_dir)
    csv_files = sorted([f for f in all_files if f.endswith(".csv")])
    yaml_files = sorted([f for f in all_files if f.endswith(".yaml")])
    
    datasets = {}
    for csv_file in csv_files:
        dataset_name = Path(csv_file).stem
        yaml_file = dataset_name + ".yaml"
        if yaml_file in yaml_files:
            csv_path = os.path.join(datasets_dir, csv_file)
            yaml_path = os.path.join(datasets_dir, yaml_file)
            
            ft_config_path = os.path.join(datasets_dir, f"{dataset_name}_ft_config.json")
            if not os.path.exists(ft_config_path):
                generate_ft_config(yaml_path, ft_config_path)
            
            datasets[dataset_name] = {
                "csv": csv_path,
                "yaml": yaml_path,
                "ft_config": ft_config_path,
            }
        else:
            print(f"No YAML descriptor found for dataset {dataset_name}. Skipping.")
    
    dataset_names = sorted(datasets.keys())
    
    for dataset_name in dataset_names:
    
        processed_dataset_folder = os.path.join(processed_sets_dir, f"{dataset_name}.bp")

        if os.path.isdir(processed_dataset_folder) and os.listdir(processed_dataset_folder):
            print(f"Dataset {dataset_name} already processed. Skipping.")
        else:
            process_dataset(datasets[dataset_name]["csv"], datasets[dataset_name]["yaml"], processed_dataset_folder)
        
        ft_config_file = datasets[dataset_name]["ft_config"]
        training_ensemble(processed_dataset_folder, dataset_name, ensemble_path, ft_config_file, model_save_dir)


if __name__ == "__main__":
    main()
