import pandas as pd

import utils.update_model as um


import os, json
from pathlib import Path

import logging
import sys

info = logging.info

import mpi4py
mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

import argparse

import hydragnn
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.distributed import get_device, get_device_name, print_peak_memory, check_remaining
from hydragnn.utils.model.model import calculate_avg_deg, Checkpoint
from hydragnn.utils.print.print_utils import print_master
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.utils.distributed import (
    setup_ddp,
    get_distributed_model,
    print_peak_memory,
)
from hydragnn.utils.input_config_parsing import update_config_edge_dim, update_config_equivariance
from hydragnn.utils.model import update_multibranch_heads
import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.train.train_validate_test import get_nbatch, get_head_indices, reduce_values_ranks, gather_tensor_ranks

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    check_if_graph_size_variable,
    gather_deg,
)
from hydragnn.utils.model import (
    save_model,
)

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None

import torch
import glob, re
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

def update_config_ensemble(model_dir_list, train_loader, val_loader, test_loader, checkpoint_dir, checkpoint_path, GFM_2024=False):
    for imodel, modeldir in enumerate(model_dir_list):
        input_filename = os.path.join(modeldir, "config.json")
        with open(input_filename, "r") as f:
            config = json.load(f)

        """check if config input consistent and update config with model and datasets"""
        graph_size_variable = os.getenv("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE")
        if graph_size_variable is None:
            graph_size_variable = check_if_graph_size_variable(
                train_loader, val_loader, test_loader
            )
        else:
            graph_size_variable = bool(int(graph_size_variable))

        if "Dataset" in config:
            check_output_dim_consistent(train_loader.dataset[0], config)

        # Set default values for GPS variables
        if "global_attn_engine" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["global_attn_engine"] = None
        if "global_attn_type" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["global_attn_type"] = None
        if "global_attn_heads" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["global_attn_heads"] = 0
        if "pe_dim" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["pe_dim"] = 0

        # update output_heads with latest config rules
        config["NeuralNetwork"]["Architecture"]["output_heads"] = update_multibranch_heads(
            config["NeuralNetwork"]["Architecture"]["output_heads"]
        )

        # This default is needed for update_config_NN_outputs
        if "compute_grad_energy" not in config["NeuralNetwork"]["Training"]:
            config["NeuralNetwork"]["Training"]["compute_grad_energy"] = False

        if GFM_2024:
            config["NeuralNetwork"]["Architecture"]["input_dim"] = len(
                config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
        )
        else:
            config["NeuralNetwork"]["Architecture"]["input_dim"] = len(
                  config["NeuralNetwork"]["Variables_of_interest"].get("input_node_features", [0])
          )
        PNA_models = ["PNA", "PNAPlus", "PNAEq"]
        if config["NeuralNetwork"]["Architecture"]["mpnn_type"] in PNA_models:
            if hasattr(train_loader.dataset, "pna_deg"):
                ## Use max neighbours used in the datasets.
                deg = torch.tensor(train_loader.dataset.pna_deg)
            else:
                deg = gather_deg(train_loader.dataset)
            config["NeuralNetwork"]["Architecture"]["pna_deg"] = deg.tolist()
            config["NeuralNetwork"]["Architecture"]["max_neighbours"] = len(deg) - 1
        else:
            config["NeuralNetwork"]["Architecture"]["pna_deg"] = None

        # Set CGCNN hidden dim to input dim if global attention is not being used
        if (
                config["NeuralNetwork"]["Architecture"]["mpnn_type"] == "CGCNN"
                and not config["NeuralNetwork"]["Architecture"]["global_attn_engine"]
        ):
            config["NeuralNetwork"]["Architecture"]["hidden_dim"] = config["NeuralNetwork"][
                "Architecture"
            ]["input_dim"]

        if config["NeuralNetwork"]["Architecture"]["mpnn_type"] == "MACE":
            if hasattr(train_loader.dataset, "avg_num_neighbors"):
                ## Use avg neighbours used in the dataset.
                avg_num_neighbors = torch.tensor(train_loader.dataset.avg_num_neighbors)
            else:
                avg_num_neighbors = float(calculate_avg_deg(train_loader.dataset))
            config["NeuralNetwork"]["Architecture"]["avg_num_neighbors"] = avg_num_neighbors
        else:
            config["NeuralNetwork"]["Architecture"]["avg_num_neighbors"] = None

        if "radius" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["radius"] = None
        if "radial_type" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["radial_type"] = None
        if "distance_transform" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["distance_transform"] = None
        if "num_gaussians" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["num_gaussians"] = None
        if "num_filters" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["num_filters"] = None
        if "envelope_exponent" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["envelope_exponent"] = None
        if "num_after_skip" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["num_after_skip"] = None
        if "num_before_skip" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["num_before_skip"] = None
        if "basis_emb_size" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["basis_emb_size"] = None
        if "int_emb_size" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["int_emb_size"] = None
        if "out_emb_size" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["out_emb_size"] = None
        if "num_radial" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["num_radial"] = None
        if "num_spherical" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["num_spherical"] = None
        if "radial_type" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["radial_type"] = None
        if "correlation" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["correlation"] = None
        if "max_ell" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["max_ell"] = None
        if "node_max_ell" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["node_max_ell"] = None

        config["NeuralNetwork"]["Architecture"] = update_config_edge_dim(
            config["NeuralNetwork"]["Architecture"]
        )

        config["NeuralNetwork"]["Architecture"] = update_config_equivariance(
            config["NeuralNetwork"]["Architecture"]
        )

        if "freeze_conv_layers" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["freeze_conv_layers"] = False
        if "initial_bias" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["initial_bias"] = None

        if "activation_function" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["activation_function"] = "relu"

        if "SyncBatchNorm" not in config["NeuralNetwork"]["Architecture"]:
            config["NeuralNetwork"]["Architecture"]["SyncBatchNorm"] = False

        if "conv_checkpointing" not in config["NeuralNetwork"]["Training"]:
            config["NeuralNetwork"]["Training"]["conv_checkpointing"] = False

        if "loss_function_type" not in config["NeuralNetwork"]["Training"]:
            config["NeuralNetwork"]["Training"]["loss_function_type"] = "mse"

        if "Optimizer" not in config["NeuralNetwork"]["Training"]:
            config["NeuralNetwork"]["Training"]["Optimizer"]["type"] = "AdamW"

        #Save config in pretrained_model_ensemble directory
        if not checkpoint_dir:
            hydragnn.utils.input_config_parsing.save_config(config, log_name="", path=modeldir)
        #Save config in finetuned_model_ensemble directory in fintetuning_log_dir
        else:
            modeldir_split = os.path.split(modeldir)[-1]
            os.makedirs(os.path.join(checkpoint_path,modeldir_split), exist_ok=True)
            hydragnn.utils.input_config_parsing.save_config(config, log_name="", path=os.path.join(checkpoint_path,modeldir_split))

def _force_dataset_name_2d(batch):
    if getattr(batch, "batch", None) is None:
        num_graphs = 1
        device = batch.x.device
    else:
        num_graphs = int(batch.batch.max().item()) + 1
        device = batch.batch.device

    ds = torch.zeros((num_graphs, 1), device=device, dtype=torch.long)
    setattr(batch, "dataset_name", ds)
    return batch
def get_distributed_model_find_unused(model, verbosity=0):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if dist.is_available() and dist.is_initialized():
        device = get_device()
        model = model.to(device)
        if device.type == "cuda":
            return DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )
        return DDP(model, find_unused_parameters=True)
    return model


def update_head_string(s: str) -> str:
    """
    Modify strings like:
        "module.heads_NN.0.0.weight"
    into:
        "module.heads_NN.0.branch-0.0.weight"
    """
    parts = s.split(".")
    # Insert "branch-0" as a separate part before the second index
    if len(parts) > 3:
        parts.insert(3, "branch-0")
        return ".".join(parts)
    return s

def update_graph_shared_string(s: str) -> str:
    """
    Modify strings like:
        "module.heads_NN.0.0.weight"
    into:
        "module.heads_NN.0.branch-0.0.weight"
    """
    parts = s.split(".")
    # Insert "branch-" before the second index
    if len(parts) > 3:
        parts[2] = f"branch-0.{parts[2]}"
        return ".".join([parts[0], parts[1], parts[2], parts[3]])
    return s


def update_GFM_2024_checkpoint(
    model, model_name, path="./logs/", optimizer=None, use_deepspeed=False, GFM_2024=False
):
    """Load both model and optimizer state from a single checkpoint file, renaming head keys.
       If <model_name>.pk is missing, fall back to the highest <model_name>_epoch_*.pk.
    """
    model_dir = os.path.join(path, model_name)
    base_pk = os.path.join(model_dir, f"{model_name}.pk")

    if os.path.isfile(base_pk):
        path_name = base_pk
    else:
        # pick highest epoch like <model>_epoch_XX.pk
        candidates = glob.glob(os.path.join(model_dir, f"{model_name}_epoch_*.pk"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found in {model_dir}: "
                f"expected {model_name}.pk or {model_name}_epoch_*.pk"
            )

        def _epoch_num(fn):
            m = re.search(r"_epoch_(\d+)\.pk$", os.path.basename(fn))
            return int(m.group(1)) if m else -1

        candidates.sort(key=_epoch_num)
        path_name = candidates[-1]

    map_location = {"cuda:%d" % 0: get_device_name()}
    print_master("Load existing model:", path_name)

    checkpoint = torch.load(path_name, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Build a renamed copy so we don't modify while iterating
    if GFM_2024:
        renamed_state_dict = {}
        for k, v in state_dict.items():
            if "head" in k:
                new_k = update_head_string(k)
            elif "graph_shared" in k:
                new_k = update_graph_shared_string(k)
            else:
                new_k = k
            renamed_state_dict[new_k] = v
        state_dict = renamed_state_dict  # write back the updated mapping
        model.load_state_dict(state_dict)

    else:
        needs_rename = any(
            (".heads_NN." in k or ".graph_shared." in k) and ("branch-" not in k)
            for k in state_dict.keys()
        )

        if needs_rename:
            renamed_state_dict = {}
            for k, v in state_dict.items():
                if ".heads_NN." in k:
                    new_k = update_head_string(k)
                elif ".graph_shared." in k:
                    new_k = update_graph_shared_string(k)
                else:
                    new_k = k
                renamed_state_dict[new_k] = v
            state_dict = renamed_state_dict

        model.load_state_dict(state_dict, strict=False)

    # Optionally load optimizer state
    if (optimizer is not None) and ("optimizer_state_dict" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class model_ensemble(torch.nn.Module):
    def __init__(self, model_dir_list, modelname=None, dir_extra="", verbosity=1, fine_tune_config = None, GFM_2024=False):
        super(model_ensemble, self).__init__()
        self.model_dir_list = model_dir_list 
        self.model_ens = torch.nn.ModuleList()
        for imodel, modeldir in enumerate(self.model_dir_list):
            input_filename = os.path.join(modeldir, "config.json")
            print("INPUT_FILENAME", input_filename, flush=True)
            with open(input_filename, "r") as f:
                config = json.load(f)
            model = hydragnn.models.create_model_config(
                config=config["NeuralNetwork"],
                verbosity=verbosity,
            )
            #model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
            model = get_distributed_model_find_unused(model, verbosity)
            # Print details of neural network architecture
            # print("Loading model %d, %s"%(imodel, modeldir))
            #if modelname is None:
            #    if not GFM_2024:
            #        hydragnn.utils.model.load_existing_model(model, os.path.basename(modeldir), path=os.path.dirname(modeldir))
            #    else:
            #        update_GFM_2024_checkpoint(model, os.path.basename(modeldir), path=os.path.dirname(modeldir), GFM_2024=GFM_2024)
            #else:
            #    if not GFM_2024:
            #        hydragnn.utils.model.load_existing_model(model, modelname, path=modeldir+dir_extra)
            #    else:
            #        update_GFM_2024_checkpoint(model, os.path.basename(modeldir), path=os.path.dirname(modeldir), GFM_2024=GFM_2024)

            if fine_tune_config is not None: 
                model = model.module
                model = um.update_model(model, fine_tune_config)
                if GFM_2024:
                    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
                else:
                    model = get_distributed_model_find_unused(model, verbosity)

            self.model_ens.append(model)

        self.num_heads = self.model_ens[0].module.num_heads
        self.head_type = self.model_ens[0].module.head_type
        self.head_dims = self.model_ens[0].module.head_dims
        self.model_size = len(self.model_dir_list)

    def forward(self, x, meanstd=False):
        y_ens=[]
        for model in self.model_ens:
            y_ens.append(model(x))
        if meanstd:
            head_pred_mean = []
            head_pred_std = []
            for ihead in range(self.num_heads):
                head_pred = []
                for imodel in range(self.model_size):
                    head_pred.append(y_ens[imodel][ihead])
                head_pred_ens = torch.stack(head_pred, dim=0).squeeze()
                head_pred_mean.append(head_pred_ens.mean(axis=0))
                head_pred_std.append(head_pred_ens.std(axis=0))
            return head_pred_mean, head_pred_std
        return y_ens

    def loss(self, pred, y, head_index):
        total_loss = 0; total_tasks_loss = [0]*len(pred)
        for k,model in enumerate(self.model_ens):
            loss,tasks_loss = model.module.loss(pred[k], y, head_index)
            total_loss += loss
            total_tasks_loss = [a + b for a, b in zip(total_tasks_loss, tasks_loss)]
        return loss, tasks_loss

    def loss_and_backprop(self, data):
        ntasks = data.y.shape[1]
        device = data.y.device
        total_tasks_loss = torch.zeros(ntasks, device=device, dtype=torch.float32)

        for model in self.model_ens:
            head_index = get_head_indices(model, data)
            pred = model(data)

            # handle DDP-wrapped models transparently
            loss_fn_owner = model.module if hasattr(model, "module") else model
            loss, tasks_loss = loss_fn_owner.loss(pred, data.y.float(), head_index)  # tasks_loss shape: [ntasks]

            model.zero_grad(set_to_none=True)
            loss.backward()

            # accumulate per-task losses (detach since it's for logging/return)
            for i, task_loss in enumerate(tasks_loss):
                total_tasks_loss[i] += task_loss.detach().to(device)

        # if you want the ensemble average per task:
        avg_tasks_loss = total_tasks_loss / len(self.model_ens)
        return avg_tasks_loss

    def val_loss(self, data):
        ntasks = data.y.shape[1]
        device = data.y.device
        total_tasks_loss = torch.zeros(ntasks, device=device, dtype=torch.float32)

        for model in self.model_ens:
            head_index = get_head_indices(model, data)
            pred = model(data)

            loss_fn_owner = model.module if hasattr(model, "module") else model
            _, tasks_loss = loss_fn_owner.loss(pred, data.y.float(), head_index)  # shape: [ntasks]

            # accumulate losses (detach: validation doesn’t need grads)
            for i, task_loss in enumerate(tasks_loss):
                total_tasks_loss[i] += task_loss.detach().to(device)

        # average over ensemble
        avg_tasks_loss = total_tasks_loss / len(self.model_ens)
        return avg_tasks_loss

    def __len__(self):
        return self.model_size


def test_ensemble(model_ens, loader, dataset_name, verbosity, num_samples=None):
    last_underscore_index = dataset_name.rfind('_')
    task = dataset_name[:last_underscore_index]

    n_ens=len(model_ens.module)
    num_heads=model_ens.module.num_heads

    num_samples_total = 0
    device = next(model_ens.parameters()).device

    true_values = [[] for _ in range(num_heads)]
    predicted_values = [[[] for _ in range(num_heads)] for _ in range(n_ens)]

    natoms =[]
   
    for data in iterate_tqdm(loader, verbosity):
        data=data.to(device)
        data = _force_dataset_name_2d(data)
        head_index = get_head_indices(model_ens.module.model_ens[0], data)
        pred_ens = model_ens(data)
        ytrue = data.y

        natoms.append(data.natoms)

        for ihead in range(num_heads):
            head_val = ytrue[head_index[ihead]]
            true_values[ihead].extend(head_val)
        for imodel, pred in enumerate(pred_ens):
            if imodel==0:
                num_samples_total += data.num_graphs
            for ihead in range(num_heads):
                head_pre = pred[ihead].reshape(-1, 1)
                pred_shape = head_pre.shape
                predicted_values[imodel][ihead].extend(head_pre.tolist())
        if num_samples is not None and num_samples_total > num_samples//hydragnn.utils.get_comm_size_and_rank()[0]-1:
            break

    predicted_mean= [[] for _ in range(num_heads)]
    predicted_std= [[] for _ in range(num_heads)]
    for ihead in range(num_heads):
        head_pred = []
        for imodel in range(len(model_ens.module)):
            head_all = torch.tensor(predicted_values[imodel][ihead])
            head_all = gather_tensor_ranks(head_all)
            # print("imodel %d"%imodel, head_all.size())
            # if debug_nan(head_all, message="pred from model %d"%imodel):
            #     print("Warning: NAN detected in model %d; prediction skipped"%imodel)
            #     continue
            head_pred.append(head_all)

        true_values[ihead] = torch.cat(true_values[ihead], dim=0)
        head_pred_ens = torch.stack(head_pred, dim=0).squeeze()
        head_pred_mean = head_pred_ens.mean(axis=0)
        head_pred_std = head_pred_ens.std(axis=0)
        true_values[ihead] = gather_tensor_ranks(true_values[ihead])
        predicted_mean[ihead] = head_pred_mean 
        predicted_std[ihead] = head_pred_std
    return (
        true_values,
        predicted_values,
        natoms,
    )

def get_model_directory(path_to_ensemble):
    modeldirlists = path_to_ensemble.split(",")
    assert len(modeldirlists) == 1 or len(modeldirlists) == 2
    if len(modeldirlists) == 1:
        modeldirlist = [os.path.join(path_to_ensemble, name) for name in os.listdir(path_to_ensemble) if
                        os.path.isdir(os.path.join(path_to_ensemble, name))]
    else:
        modeldirlist = []
        for models_dir_folder in modeldirlists:
            modeldirlist.extend([os.path.join(models_dir_folder, name) for name in os.listdir(models_dir_folder) if
                                 os.path.isdir(os.path.join(models_dir_folder, name))])

    var_config = None
    for modeldir in modeldirlist:
        input_filename = os.path.join(modeldir, "config.json")
        with open(input_filename, "r") as f:
            config = json.load(f)
        if var_config is not None:
            assert var_config == config["NeuralNetwork"][
                "Variables_of_interest"], "Inconsistent variable config in %s" % input_filename
        else:
            var_config = config["NeuralNetwork"]["Variables_of_interest"]
    verbosity = config["Verbosity"]["level"]

def train_ensemble(model_ensemble, train_loader, val_loader, num_epochs, optimizers, device="cpu", member_checkpoints=None, GFM_2024=False):
    """
    Train an ensemble of models using a custom loss_and_backprop function.

    Parameters:
        model_ensemble: A model object containing the ensemble (with a `loss_and_backprop` method).
        dataloader: PyTorch DataLoader providing (input, target) batches.
        num_epochs: Number of training epochs.
        optimizers: List of optimizers, one for each model in the ensemble.
        device: Device to use ("cpu" or "cuda").
    """
    finetuning_log_dir = os.getenv("FINETUNING_LOG_DIR")
    for epoch in iterate_tqdm(
                 range(num_epochs), verbosity_level=2, desc="Epoch", total=num_epochs
        ):
        # expose epoch to checkpoint file naming
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)
        model_ensemble.train()  # Set ensemble to training mode
        epoch_loss = 0  # Track cumulative loss for the epoch
        nbatch = get_nbatch(train_loader)
        for ibatch, batch in iterate_tqdm(
                    enumerate(train_loader), verbosity_level=2, desc="Train Ensemble", total=nbatch
            ):
            # Forward pass and backpropagation
            if not GFM_2024:
                batch = batch.to(get_device())
                batch = _force_dataset_name_2d(batch)
            total_tasks_loss = torch.atleast_1d(model_ensemble.module.loss_and_backprop(batch))

            # Optimizer step for each ensemble member
            for optimizer in optimizers:
                optimizer.step()

            # Optionally accumulate loss for logging
            epoch_loss += total_tasks_loss[0]

        mean_train_loss_epoch = epoch_loss/len(train_loader)

        #validation 
        model_ensemble.eval()
        val_loss = 0 
        for batch in val_loader: 
            #forward pass 
            if not GFM_2024:
                batch = batch.to(get_device())
                batch = _force_dataset_name_2d(batch)
            total_tasks_loss = torch.atleast_1d(model_ensemble.module.val_loss(batch.to(get_device())))

            #accumulate validation loss for logging 
            val_loss += total_tasks_loss[0]

        mean_val_loss_epoch = val_loss/len(val_loader)
    
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_train_loss_epoch:.4f}, Val Loss: {mean_val_loss_epoch:.4f}")

        # checkpoint each fine-tuned member when validation improves
        if member_checkpoints:
            perf = float(mean_val_loss_epoch.detach().cpu())
            for i, cp in enumerate(member_checkpoints):
                if cp(model_ensemble.module.model_ens[i], optimizers[i], perf):
                    print_master("Creating Checkpoint: %f" % cp.min_perf_metric)
                print_master("Best Performance Metric: %f" % cp.min_perf_metric)


def run_finetune(dictionary_variables, args):
    """Main training/validation/test entry point.
    Accepts an argparse.Namespace-like `args`.
    """
    modelname = "FineTuning" if args.modelname is None else args.modelname
    datasetname = "FineTuning" if args.datasetname is None else args.datasetname
    gfm_2024 = args.gfm_2024

    # ---- tracing / timers ----
    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    # ---- load fine-tuning config ----
    ftcfgfile = args.finetuning_config
    print(ftcfgfile, flush=True)
    with open(ftcfgfile, "r") as f:
        ft_config = json.load(f)

    # ---- default FINETUNING_LOG_DIR if not set ----
    finetuning_log_dir = os.getenv("FINETUNING_LOG_DIR")
    if not finetuning_log_dir:
        try:
            example_dir = Path(os.path.abspath(ftcfgfile)).parent
            finetuning_log_dir = str(example_dir / "logs")
        except Exception:
            finetuning_log_dir = "./logs"
        os.environ["FINETUNING_LOG_DIR"] = finetuning_log_dir
    os.makedirs(finetuning_log_dir, exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(os.path.join(finetuning_log_dir, modelname), exist_ok=True)

    verbosity = ft_config["Verbosity"]["level"]
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = dictionary_variables['graph_feature_names']
    var_config["graph_feature_dims"] = dictionary_variables['graph_feature_dims']
    var_config["node_feature_names"] = dictionary_variables['node_feature_names']
    var_config["node_feature_dims"] = dictionary_variables['node_feature_dims']

    if not gfm_2024:
        var_config["input_node_features"] = [0]

    if args.batch_size is not None:
        ft_config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if args.num_epochs is not None:
        ft_config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epochs

    # ---- initialize DDP / MPI ----
    comm_size, rank = setup_ddp()
    comm = MPI.COMM_WORLD

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = modelname
    hydragnn.utils.print.print_utils.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name, path=finetuning_log_dir)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    # ---- dataset loading ----
    if args.format == "adios":
        info("Adios load")
        if AdiosDataset is None:
            raise ImportError("AdiosDataset not available. Reinstall with ADIOS2 support.")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), f"./dataset/{datasetname}.bp")
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset   = AdiosDataset(fname, "valset",   comm, **opt, var_config=var_config)
        testset  = AdiosDataset(fname, "testset",  comm, **opt, var_config=var_config)

    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(os.path.dirname(__file__), "../dataset", f"{datasetname}.pickle")
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
        valset   = SimplePickleDataset(basedir=basedir, label="valset",   var_config=var_config)
        testset  = SimplePickleDataset(basedir=basedir, label="testset",  var_config=var_config)

        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset   = DistDataset(valset,   "valset",   comm, **opt)
            testset  = DistDataset(testset,  "testset",  comm, **opt)
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError(f"No supported format: {args.format}")

    print("Loaded dataset.")
    info("trainset,valset,testset size: %d %d %d" % (len(trainset), len(valset), len(testset)))

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
    (train_loader, val_loader, test_loader) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, ft_config["NeuralNetwork"]["Training"]["batch_size"]
    )

    # Prepare per-member checkpoints if enabled
    member_checkpoints = None
    try:
        save_ckpt = ft_config["NeuralNetwork"]["Training"].get("Checkpoint", False)
    except Exception:
        save_ckpt = False

    if save_ckpt:
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(finetuning_log_dir, modelname)
        else:
            checkpoint_path = finetuning_log_dir

    # ---- ensemble setup ----
    ensemble_path = Path(args.pretrained_model_ensemble_path)
    model_dir_list = [os.path.join(ensemble_path, model_id) for model_id in os.listdir(ensemble_path)]
    update_config_ensemble(model_dir_list, train_loader, val_loader, test_loader, args.checkpoint_dir, checkpoint_path)

    model = model_ensemble(model_dir_list, fine_tune_config=ft_config, GFM_2024=gfm_2024)
    if gfm_2024:
        model = get_distributed_model(model, verbosity=2)
    else:
        model = get_distributed_model_find_unused(model, verbosity=2)

    print("Created Dataloaders")
    comm.Barrier()
    timer.stop()

    # ---- optimizers, checkpoints, train, test ----
    optimizers = [
        torch.optim.Adam(
            member.parameters(),
            lr=ft_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"],
        )
        for member in model.module.model_ens
    ]


    if save_ckpt:
        warmup = ft_config["NeuralNetwork"]["Training"].get("checkpoint_warmup", 0)
        member_checkpoints = []
        for idx, _ in enumerate(model.module.model_ens):
            # Use the original pretrained model directory name
            member_name = os.path.basename(model_dir_list[idx])
            # ensure log directory exists for this member under configured logs root
            if args.checkpoint_dir:
                os.makedirs(os.path.join(finetuning_log_dir, modelname, member_name), exist_ok=True)
                member_checkpoints.append(
                    Checkpoint(
                        name=member_name,
                        warmup=warmup,
                        path=os.path.join(finetuning_log_dir, modelname),
                        use_deepspeed=False,
                    )
                )
            else:
                os.makedirs(os.path.join(finetuning_log_dir, member_name), exist_ok=True)
                member_checkpoints.append(
                    Checkpoint(
                        name=member_name,
                        warmup=warmup,
                        path=finetuning_log_dir,
                        use_deepspeed=False,
                    )
                )

    train_ensemble(
        model,
        train_loader,
        val_loader,
        num_epochs=ft_config["NeuralNetwork"]["Training"]["num_epoch"],
        optimizers=optimizers,
        device="cuda",
        member_checkpoints=member_checkpoints,
        GFM_2024=gfm_2024,
    )

    # Test the ensemble
    true_, pred, natoms = test_ensemble(model, test_loader, modelname, verbosity=2)

    last_underscore_index = datasetname.rfind('_')
    task = datasetname[:last_underscore_index]


    for i in range(len(natoms)):
        if i == 0:
            natoms_ = natoms[i]
        else:
            natoms_ = torch.cat((natoms_, natoms[i]),dim=0)

    pred_ = []
    for i in range(len(pred[0][0])):
        pred_.append(pred[0][0][i])
    pred_ = torch.tensor(pred_).squeeze()

    if task in ["matbench_jdft2d"]:

        print("PRED", pred_.to('cpu')*1000/natoms_.to('cpu'))
        print("TRUE", true_[0].to('cpu')*1000/natoms_.to('cpu'))
        absdiff = torch.absolute(true_[0].to('cpu')-pred_.to('cpu'))
        absdiff_scaled = torch.absolute(true_[0].to('cpu')*1000/natoms_.to('cpu')-pred_.to('cpu')*1000/natoms_.to('cpu'))
        print("absdiff_scaled", absdiff_scaled)
        print("MAE_scaled", torch.mean(absdiff_scaled))

        print("RMSE", torch.sqrt(torch.mean((true_[0].to('cpu')*1000/natoms_.to('cpu')-pred_.to('cpu')*1000/natoms_.to('cpu'))**2)))
        print("Max", torch.max(absdiff_scaled))

    elif task in ["matbench_mp_is_metal"]:
        print("CORRECT_COUNT: ", torch.eq(true_[0].to('cpu'), (torch.sigmoid(pred_.to('cpu'))> 0.5)).sum().item())
        print("Percentage Correct: ", torch.eq(true_[0].to('cpu'), (torch.sigmoid(pred_.to('cpu'))> 0.5)).sum().item() / true_[0].to('cpu').shape[0])
        print("ROCAUC: ", roc_auc_score(true_[0].detach().cpu().numpy(), pred_.detach().cpu().numpy()))
        print("F1: ", f1_score(true_[0].detach().cpu().numpy(), (torch.sigmoid(pred_)>0.5).detach().cpu().numpy()))
        print("Balanced_acc: ", balanced_accuracy_score(true_[0].detach().cpu().numpy(), (torch.sigmoid(pred_)>0.5).detach().cpu().numpy()))

    # Optional: return something useful (e.g., final metrics)
    return True


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
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
    parser.add_argument("--datasetname", help="dataset name")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--num_epochs", type=int, help="num_epoch", default=None)
    parser.add_argument("--checkpoint_dir", action='store_true', help="Whether to store checkpoint within a folder based on modelname, useful for case where multiple models are finetuned, e.g. matbench")
    parser.add_argument("--gfm_2024", action='store_true', help="Whether to use 2024 gfm model or not")
    parser.add_argument("--freeze", action='store_true', help="freeze layers or not")

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
    return parser
