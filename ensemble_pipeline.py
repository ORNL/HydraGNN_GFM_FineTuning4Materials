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
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import nsplit
import sys
sys.path.append('/Users/varrunprakash/Desktop/Finetuing/GFM_FineTune/data_utils')

from data_utils.models import DataDescriptor, number_categories

from hydragnn.utils.distributed import setup_ddp, get_distributed_model
from hydragnn.utils.time_utils import Timer
from update_model import update_model, update_ensemble
from ensemble_utils import model_ensemble, train_ensemble, test_ensemble
import torch.distributed as dist


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("hydragnn").setLevel(logging.WARNING)
logging.getLogger("adios2").setLevel(logging.WARNING)

def random_splits(N, train_frac, val_frac):
   
    assert 0.0 <= train_frac <= 1.0
    assert 0.0 <= val_frac <= 1.0 - train_frac
    indices = list(range(N))
    indices = random.sample(indices, N)
    return np.split(indices, [int(train_frac * N), int((train_frac + val_frac) * N)])


def validate_data(x: np.ndarray, ncat: int, tol=0.001) -> None:
    if ncat == 0:  
        return
    assert ncat != 1, "Invalid number of categories."
    neg_one = np.abs(x + 1.0) < tol
    x[neg_one] = np.nan
    x = x[~np.isnan(x)]
    y = x.astype(int)
    assert np.allclose(x, y, atol=tol)
    assert np.all(x >= 0), "Negative categorical values are not allowed."
    assert np.all(x < ncat), "Categorical values out of range."


def validate_split_names(split_series):
    valid_names = ["train", "val", "test", "excl"]
    valid = split_series.str.startswith(tuple(valid_names))
    if valid.all():
        return
    else:
        print("Mismatch in split names. The following rows have invalid splits:")
        print(split_series[~valid])


def load_columns(datafile: str, descr):
   
    if datafile.endswith("csv"):
        df = pd.read_csv(datafile)
    else:
        df = pd.read_parquet(datafile)
    smiles_all = df[descr.smiles].to_list()
    names = [val.name for val in descr.graph_tasks]
    values_all = df[names].values.astype(float)
    N = len(smiles_all)
    print("    total records:", N)
    for i, task in enumerate(descr.graph_tasks):
        ncat = number_categories(task.type)
        validate_data(values_all[:, i], ncat)

    if descr.split is None:
       
        idxs = random_splits(N, 0.8, 0.1)
    else:
        split = df[descr.split]
        validate_split_names(split)
        idxs = [np.flatnonzero(split.str.startswith(name)) for name in ["train", "val", "test"]]
    smiles = [[smiles_all[i] for i in ix] for ix in idxs]
    values = [torch.tensor([values_all[i] for i in ix]) for ix in idxs]
    return smiles, values, names


def calc_offsets(counts):
    off = []
    total = 0
    for count in counts:
        off.append(total)
        total += count
    return off


def group_features(tasks):
   
    names = []
    sizes = []
    types = []
    name = ""
    sz = 0
    cur_type = None

    def finalize(v):
        nonlocal names, sizes, types, name, sz, cur_type
        if sz > 0:
            names.append(name)
            sizes.append(sz)
            types.append(cur_type)
        if v is None:
            return
        name = v["name"]
        sz = 1
        cur_type = v["type"]

    start = True
    for v in tasks:
        if start:
            finalize(v)
            start = False
        elif v["type"] != cur_type:
            finalize(v)
        else:
            name += " " + v["name"]
            sz += 1

    finalize(None)
    return names, sizes, types


def get_architecture_config(descr):
    group_names, group_sizes, _ = group_features(descr["graph_tasks"])
    ntasks = sum(group_sizes)
    arc_config = {
        "output_heads": {
            "graph": {
                "dim_pretrained": 50,
                "num_headlayers": 2,
                "dim_headlayers": [50, 25],
            }
        },
        "task_weights": [1.0] * ntasks,
        "output_dim": [1] * ntasks,
        "output_type": ["graph"],
    }
    return arc_config


def get_training_config(descr):
   
    loss_fun_dict = descr['graph_tasks'][0]
    return {
        "num_epoch": 3,
        "EarlyStopping": True,
        "perc_train": 0.9,
        "loss_function_types": loss_fun_dict,
        "batch_size": 32,
        "continue": 0,
        "Optimizer": {"type": "AdamW", "learning_rate": 1e-05},
        "conv_checkpointing": False,
    }


def generate_ft_config(yaml_file, output_config_file):
  
    with open(yaml_file, "r", encoding="utf-8") as f:
        descr = yaml.safe_load(f)
    arc_config = get_architecture_config(descr)
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
        from data_utils.models import DataDescriptor
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
                from hydragnn.utils.smiles_utils import generate_graphdata_from_smilestr
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
        logging.print(f"Total missed molecules for dataset {dataset_name}: {missed_total}")
    
    from hydragnn.preprocess.utils import gather_deg
    pna_deg = gather_deg(dataset_data["trainset"])

    from hydragnn.utils.smiles_utils import get_node_attribute_name, get_edge_attribute_name
    node_names, node_dims = get_node_attribute_name()
    node_names = [x.replace("atomicnumber", "atomic_number") for x in node_names]
    edge_names, edge_dims = get_edge_attribute_name()
    task_dims = [1] * len(task_names)

    def calc_offsets(counts):
        off = []
        total = 0
        for count in counts:
            off.append(total)
            total += count
        return off

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

    from hydragnn.utils.adiosdataset import AdiosWriter
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

    import hydragnn.utils.tracer as tr
    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    world_size, world_rank = setup_ddp()
    comm = MPI.COMM_WORLD
   
    dataset_file = str(Path(processed_dataset_dir))

    model_files = [os.path.join(ensemble_path, model_id) for model_id in os.listdir(ensemble_path)]
    from ensemble_utils import model_ensemble
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
    
    
    from hydragnn.utils.adiosdataset import AdiosDataset
    trainset = AdiosDataset(dataset_file, "trainset", comm, **opt)
    valset = AdiosDataset(dataset_file, "valset", comm, **opt)
    testset = AdiosDataset(dataset_file, "testset", comm, **opt)
    
    print("Loaded dataset for training.")
    print("trainset,valset,testset sizes: %d %d %d", len(trainset), len(valset), len(testset))
    
    
    from hydragnn.preprocess import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        trainset, valset, testset, ft_config["Training"]["batch_size"]
    )
    print("Created Dataloaders.")
    comm.Barrier()
    
    timer.stop()
    
    optimizers = [torch.optim.Adam(member.parameters(), lr=ft_config["Training"]["Optimizer"]["learning_rate"])
                  for member in model.module.model_ens]
    
    from ensemble_utils import train_ensemble, test_ensemble
    for epoch in range(1, 4):
        print(f"Training dataset {dataset_name}, epoch {epoch}...")
        train_ensemble(model, train_loader, val_loader, num_epochs=1, optimizers=optimizers, device="cuda")
        
        save_dir = Path(model_save_dir) / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"model_epoch_{epoch}.pt"
        if world_rank == 0:
            torch.save(model.module.state_dict(), str(checkpoint_path))
            print(f"Saved model checkpoint to {checkpoint_path}")

    test_ensemble(model, test_loader, verbosity=2)


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
            logging.warning(f"No YAML descriptor found for dataset {dataset_name}. Skipping.")
    
    dataset_names = sorted(datasets.keys())
    
    for dataset_name in dataset_names:
    
        processed_dataset_folder = os.path.join(processed_sets_dir, f"{dataset_name}.bp")

       
        if os.path.isdir(processed_dataset_folder) and os.listdir(processed_dataset_folder):
            print(
                f"Dataset {dataset_name} already processed. "
                f"Skipping processing. Folder contents: {os.listdir(processed_dataset_folder)}"
            )
        else:
            process_dataset(
                datasets[dataset_name]["csv"],
                datasets[dataset_name]["yaml"],
                processed_dataset_folder
            )
        
      
        ft_config_file = datasets[dataset_name]["ft_config"]
        training_ensemble(
            processed_dataset_folder, 
            dataset_name, 
            ensemble_path, 
            ft_config_file, 
            model_save_dir
        )


if __name__ == "__main__":
    main()
