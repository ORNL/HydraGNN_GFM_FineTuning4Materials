import pandas as pd
import numpy as np
from typing import List

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

from hydragnn.train.train_validate_test import (
    gather_tensor_ranks,
    get_autocast_and_scaler,
    get_head_indices,
    get_nbatch,
    reduce_values_ranks,
    resolve_precision,
)

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    check_if_graph_size_variable,
    gather_deg,
)

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    AdiosWriter = None
    AdiosDataset = None

import torch
import glob, re


def _get_param_dtype(training_config: dict) -> torch.dtype:
    """Resolve the parameter dtype configured for fine-tuning."""
    _, param_dtype, _ = resolve_precision(training_config.get("precision", "fp32"))
    return param_dtype


def _get_training_autocast(training_config: dict):
    """Return the autocast context configured for fine-tuning precision."""
    autocast_context, _ = get_autocast_and_scaler(
        training_config.get("precision", "fp32")
    )
    return autocast_context


def _move_batch_to_training_precision(data, training_config: dict):
    """Move a batch to the active device and configured floating-point dtype."""
    param_dtype = _get_param_dtype(training_config)
    data = data.to(get_device())
    for field_name in data.keys():
        field_value = getattr(data, field_name, None)
        if torch.is_tensor(field_value) and torch.is_floating_point(field_value):
            setattr(data, field_name, field_value.to(dtype=param_dtype))
    return data


def _get_num_atoms_per_graph(
    data: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return the atom count for each graph in a batched PyG object."""
    count_dtype = dtype if dtype is not None else data.y.dtype
    if getattr(data, "ptr", None) is not None:
        return (data.ptr[1:] - data.ptr[:-1]).to(dtype=count_dtype, device=data.y.device)

    if getattr(data, "batch", None) is not None:
        num_graphs = int(data.batch.max().item()) + 1
        counts = torch.bincount(data.batch, minlength=num_graphs)
        return counts.to(dtype=count_dtype, device=data.y.device)

    return torch.tensor([float(data.x.shape[0])], dtype=count_dtype, device=data.y.device)


def _get_training_targets(data, training_config: dict) -> torch.Tensor:
    """Build the target tensor used for loss/evaluation from the stored dataset values."""
    target_mode = training_config.get("energy_target_mode", "per_atom")
    target_dtype = _get_param_dtype(training_config)
    y = data.y.to(dtype=target_dtype)

    if target_mode == "per_atom":
        return y

    if target_mode == "total":
        num_atoms = _get_num_atoms_per_graph(data, dtype=target_dtype).view(-1, 1)
        return y * num_atoms

    raise ValueError(
        f"Unsupported NeuralNetwork.Training.energy_target_mode={target_mode!r}. "
        "Expected 'per_atom' or 'total'."
    )


def _convert_predictions_to_per_atom(pred, data):
    """Convert graph-level total-energy predictions to per-atom values for reporting."""
    converted_pred = []
    for head_pred in pred:
        num_atoms = _get_num_atoms_per_graph(data, dtype=head_pred.dtype).view(-1, 1)
        converted_pred.append(head_pred / num_atoms)
    return converted_pred

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
    parser.add_argument(
        "--train_from_scratch",
        action="store_true",
        help="skip loading pretrained checkpoints and initialize the fine-tuning model weights from scratch",
    )

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

def update_config_ensemble(model_dir_list, train_loader, val_loader, test_loader):
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

        hydragnn.utils.input_config_parsing.save_config(config, log_name="", path=modeldir)

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
    model, model_name, path="./logs/", optimizer=None, use_deepspeed=False
):
    model_dir = os.path.join(path, model_name)
    base_pk = os.path.join(model_dir, f"{model_name}.pk")

    if os.path.isfile(base_pk):
        path_name = base_pk
    else:
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

    if (optimizer is not None) and ("optimizer_state_dict" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

class model_ensemble(torch.nn.Module):
    def __init__(
        self,
        model_dir_list,
        modelname=None,
        dir_extra="",
        verbosity=1,
        fine_tune_config=None,
        GFM_2024=False,
        train_from_scratch=False,
    ):
        super(model_ensemble, self).__init__()
        self.model_dir_list = model_dir_list
        self.training_config = (
            fine_tune_config.get("NeuralNetwork", {}).get("Training", {})
            if fine_tune_config is not None
            else {}
        )
        self.model_ens = torch.nn.ModuleList()
        for imodel, modeldir in enumerate(self.model_dir_list):
            input_filename = os.path.join(modeldir, "config.json")
            with open(input_filename, "r") as f:
                config = json.load(f)
            model = hydragnn.models.create_model_config(
                config=config["NeuralNetwork"],
                verbosity=verbosity,
            )
            model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
            if not train_from_scratch:
                if modelname is None:
                    if not GFM_2024:
                        hydragnn.utils.model.load_existing_model(model, os.path.basename(modeldir), path=os.path.dirname(modeldir))
                    else:
                        update_GFM_2024_checkpoint(model, os.path.basename(modeldir), path=os.path.dirname(modeldir))
                else:
                    if not GFM_2024:
                        hydragnn.utils.model.load_existing_model(model, modelname, path=modeldir+dir_extra)
                    else:
                        update_GFM_2024_checkpoint(model, os.path.basename(modeldir), path=os.path.dirname(modeldir))

            if fine_tune_config is not None:
                model = model.module
                model = um.update_model(model, fine_tune_config)
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
        target_y = _get_training_targets(data, self.training_config)
        per_atom_target_y = data.y.to(dtype=target_y.dtype)
        ntasks = target_y.shape[1]
        device = target_y.device
        total_tasks_loss = torch.zeros(ntasks, device=device, dtype=target_y.dtype)
        total_per_atom_tasks_loss = torch.zeros(ntasks, device=device, dtype=target_y.dtype)
        report_per_atom = self.training_config.get("energy_target_mode", "per_atom") == "total"
        for model in self.model_ens:
            head_index = get_head_indices(model, data)
            loss_fn_owner = model.module if hasattr(model, "module") else model

            # Run forward/loss under the configured autocast mode when enabled.
            with _get_training_autocast(self.training_config):
                pred = model(data)
                loss, tasks_loss = loss_fn_owner.loss(
                    pred, target_y, head_index
                )  # tasks_loss shape: [ntasks]

            model.zero_grad(set_to_none=True)
            loss.backward()

            # accumulate per-task losses (detach since it's for logging/return)
            for i, task_loss in enumerate(tasks_loss):
                total_tasks_loss[i] += task_loss.detach().to(device)

            if report_per_atom:
                per_atom_pred = _convert_predictions_to_per_atom(pred, data)
                with _get_training_autocast(self.training_config):
                    _, per_atom_tasks_loss = loss_fn_owner.loss(
                        per_atom_pred, per_atom_target_y, head_index
                    )
                for i, task_loss in enumerate(per_atom_tasks_loss):
                    total_per_atom_tasks_loss[i] += task_loss.detach().to(device)

        # if you want the ensemble average per task:
        avg_tasks_loss = total_tasks_loss / len(self.model_ens)
        avg_per_atom_tasks_loss = total_per_atom_tasks_loss / len(self.model_ens)
        return avg_tasks_loss, avg_per_atom_tasks_loss

    def val_loss(self, data):
        target_y = _get_training_targets(data, self.training_config)
        per_atom_target_y = data.y.to(dtype=target_y.dtype)
        ntasks = target_y.shape[1]
        device = target_y.device
        total_tasks_loss = torch.zeros(ntasks, device=device, dtype=target_y.dtype)
        total_per_atom_tasks_loss = torch.zeros(ntasks, device=device, dtype=target_y.dtype)
        report_per_atom = self.training_config.get("energy_target_mode", "per_atom") == "total"
        for model in self.model_ens:
            head_index = get_head_indices(model, data)
            loss_fn_owner = model.module if hasattr(model, "module") else model
            with _get_training_autocast(self.training_config):
                pred = model(data)
                _, tasks_loss = loss_fn_owner.loss(
                    pred, target_y, head_index
                )  # shape: [ntasks]

            # accumulate losses (detach: validation doesn’t need grads)
            for i, task_loss in enumerate(tasks_loss):
                total_tasks_loss[i] += task_loss.detach().to(device)

            if report_per_atom:
                per_atom_pred = _convert_predictions_to_per_atom(pred, data)
                with _get_training_autocast(self.training_config):
                    _, per_atom_tasks_loss = loss_fn_owner.loss(
                        per_atom_pred, per_atom_target_y, head_index
                    )
                for i, task_loss in enumerate(per_atom_tasks_loss):
                    total_per_atom_tasks_loss[i] += task_loss.detach().to(device)

        # average over ensemble
        avg_tasks_loss = total_tasks_loss / len(self.model_ens)
        avg_per_atom_tasks_loss = total_per_atom_tasks_loss / len(self.model_ens)
        return avg_tasks_loss, avg_per_atom_tasks_loss

    def __len__(self):
        return self.model_size

def test_ensemble_jamie(model_ens, loader, dataset_name, verbosity, save_results:bool=False, num_samples:int=None)->List[np.ndarray]:
    n_ens = len(model_ens.module)
    num_heads = model_ens.module.num_heads
    num_samples_total = 0
    device = next(model_ens.parameters()).device

    true_values = [[] for _ in range(num_heads)]
    predicted_values = [[[] for _ in range(num_heads)] for _ in range(n_ens)]

    model_ens.eval()

    with torch.no_grad():
        for data in iterate_tqdm(loader, verbosity):
            data = _move_batch_to_training_precision(data, model_ens.module.training_config)
            head_index = get_head_indices(model_ens.module.model_ens[0], data)
            with _get_training_autocast(model_ens.module.training_config):
                pred_ens = model_ens(data)
            ytrue = _get_training_targets(data, model_ens.module.training_config)

            for ihead in range(num_heads):
                head_val = ytrue[head_index[ihead]]
                true_values[ihead].extend(head_val.tolist()) # Convert to list immediately

            for imodel, pred in enumerate(pred_ens):
                if imodel == 0:
                    num_samples_total += data.num_graphs
                for ihead in range(num_heads):
                    # Flatten to ensure 1D
                    head_pre = pred[ihead].reshape(-1)
                    predicted_values[imodel][ihead].extend(head_pre.tolist())

            if num_samples is not None and num_samples_total > num_samples//hydragnn.utils.get_comm_size_and_rank()[0]-1:
                break

        final_true = []
        final_mean = []
        final_std = []

        for ihead in range(num_heads):
            head_pred_list = []
            for imodel in range(n_ens):
                # Ensure we are dealing with a 1D tensor [Samples]
                model_pred_head = torch.tensor(predicted_values[imodel][ihead], device=device).flatten()
                model_pred_head = gather_tensor_ranks(model_pred_head)
                head_pred_list.append(model_pred_head)

            # Stack to [NumModels, Samples]
            head_pred_ens = torch.stack(head_pred_list, dim=0)

            # Mean and Std across model dimension (dim=0)
            h_mean = head_pred_ens.mean(dim=0)
            if n_ens > 1:
                h_std = head_pred_ens.std(dim=0)
            else:
                h_std = torch.zeros_like(h_mean)

            # Process True values
            t_val = torch.tensor(true_values[ihead], device=device).flatten()
            t_val = gather_tensor_ranks(t_val)

            # Store as clean 1D numpy arrays
            final_true.append(t_val.cpu().numpy())
            final_mean.append(h_mean.cpu().numpy())
            final_std.append(h_std.cpu().numpy())

    if (save_results):
        outputs_df = pd.DataFrame({
            "True Value": final_true[0],
            "Predicted Value": final_mean[0],
            "Std Dev": final_std[0]
        })

    if (dataset_name is not None):
        print(f"Done testing for {dataset_name}")

    return final_true, final_mean, final_std

def test_ensemble(model_ens, loader, dataset_name, verbosity, save_results:bool=False, num_samples:int=None):
    n_ens=len(model_ens.module)
    num_heads=model_ens.module.num_heads

    num_samples_total = 0
    device = next(model_ens.parameters()).device

    true_values = [[] for _ in range(num_heads)]
    predicted_values = [[[] for _ in range(num_heads)] for _ in range(n_ens)]
    # Set to eval mode
    model_ens.eval()

    # Don't track gradients
    with torch.no_grad():
        # Iterate over dataloader
        for data in iterate_tqdm(loader, verbosity):
            data = _move_batch_to_training_precision(data, model_ens.module.training_config)
            data = _force_dataset_name_2d(data)
            head_index = get_head_indices(model_ens.module.model_ens[0], data)
            ###########################
            with _get_training_autocast(model_ens.module.training_config):
                pred_ens = model_ens(data)
            ###########################
            ytrue = _get_training_targets(data, model_ens.module.training_config)
            for ihead in range(num_heads):
                head_val = ytrue[head_index[ihead]]
                true_values[ihead].extend(head_val)
            ###########################
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
                # Convert the list of lists [[v1], [v2]] into a tensor [N, 1]
                head_all = torch.tensor(predicted_values[imodel][ihead])
                head_all = gather_tensor_ranks(head_all)
                head_pred.append(head_all)

            # true_values: shape [NumSamples, 1] -> flatten to [NumSamples]
            true_values[ihead] = torch.cat(true_values[ihead], dim=0).flatten()

            # head_pred_ens shape: [NumModels, NumSamples, 1]
            head_pred_ens = torch.stack(head_pred, dim=0)

            # Squeeze only the last dimension (the -1)
            head_pred_ens = head_pred_ens.squeeze(-1)

            # Calculate mean and std across the model dimension (dim=0)
            head_pred_mean = head_pred_ens.mean(dim=0)
            head_pred_std = head_pred_ens.std(dim=0)

            # Re-sync true values from all ranks
            true_values[ihead] = gather_tensor_ranks(true_values[ihead])

            predicted_mean[ihead] = head_pred_mean
            predicted_std[ihead] = head_pred_std

    if (save_results):
        #save values to pandas dataframe
        Smiles = "blank"
        t_values = true_values[0].tolist()
        pred_values = predicted_mean[0].tolist()
        outputs_df = pd.DataFrame({
            "SMILE Strings": Smiles,
            "True Value": t_values,
            "Predicted Value": pred_values
        })

    print(f"Done testing for {dataset_name}")

    return (
        true_values,
        predicted_mean,
        predicted_std
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

def train_ensemble(model_ensemble, train_loader, val_loader, num_epochs, optimizers, device="cpu", member_checkpoints=None):
    """
    Train an ensemble of models using a custom loss_and_backprop function.

    Parameters:
        model_ensemble: A model object containing the ensemble (with a `loss_and_backprop` method).
        dataloader: PyTorch DataLoader providing (input, target) batches.
        num_epochs: Number of training epochs.
        optimizers: List of optimizers, one for each model in the ensemble.
        device: Device to use ("cpu" or "cuda").
    """
    for epoch in iterate_tqdm(
                 range(num_epochs), verbosity_level=2, desc="Epoch", total=num_epochs
        ):
        # expose epoch to checkpoint file naming
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)
        model_ensemble.train()  # Set ensemble to training mode
        epoch_loss = 0  # Track cumulative loss for the epoch
        epoch_per_atom_loss = 0
        nbatch = get_nbatch(train_loader)
        for ibatch, batch in iterate_tqdm(
                    enumerate(train_loader), verbosity_level=2, desc="Train Ensemble", total=nbatch
            ):
            # Forward pass and backpropagation
            batch = _move_batch_to_training_precision(
                batch, model_ensemble.module.training_config
            )
            batch = _force_dataset_name_2d(batch)
            total_tasks_loss, total_per_atom_tasks_loss = model_ensemble.module.loss_and_backprop(batch)
            total_tasks_loss = torch.atleast_1d(total_tasks_loss)
            total_per_atom_tasks_loss = torch.atleast_1d(total_per_atom_tasks_loss)
            # Optimizer step for each ensemble member
            for optimizer in optimizers:
                optimizer.step()

            # Optionally accumulate loss for logging
            epoch_loss += total_tasks_loss[0]
            epoch_per_atom_loss += total_per_atom_tasks_loss[0]

        mean_train_loss_epoch = epoch_loss/len(train_loader)
        mean_train_per_atom_loss_epoch = epoch_per_atom_loss/len(train_loader)

        #validation
        model_ensemble.eval()
        val_loss = 0
        val_per_atom_loss = 0
        for batch in val_loader:
            #forward pass
            batch = _move_batch_to_training_precision(
                batch, model_ensemble.module.training_config
            )
            batch = _force_dataset_name_2d(batch)
            total_tasks_loss, total_per_atom_tasks_loss = model_ensemble.module.val_loss(batch)
            total_tasks_loss = torch.atleast_1d(total_tasks_loss)
            total_per_atom_tasks_loss = torch.atleast_1d(total_per_atom_tasks_loss)
            #accumulate validation loss for logging
            val_loss += total_tasks_loss[0]
            val_per_atom_loss += total_per_atom_tasks_loss[0]

        mean_val_loss_epoch = val_loss/len(val_loader)
        mean_val_per_atom_loss_epoch = val_per_atom_loss/len(val_loader)

        if model_ensemble.module.training_config.get("energy_target_mode", "per_atom") == "total":
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train MAE: {mean_train_loss_epoch:.4f} eV, "
                f"Val MAE: {mean_val_loss_epoch:.4f} eV, "
                f"Train MAE/atom: {mean_train_per_atom_loss_epoch:.4f} eV/atom, "
                f"Val MAE/atom: {mean_val_per_atom_loss_epoch:.4f} eV/atom"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train MAE: {mean_train_loss_epoch:.4f}, "
                f"Val MAE: {mean_val_loss_epoch:.4f}"
            )

        # checkpoint each fine-tuned member when validation improves
        if member_checkpoints:
            perf = float(mean_val_loss_epoch.detach().cpu())
            for i, cp in enumerate(member_checkpoints):
                if cp(model_ensemble.module.model_ens[i], optimizers[i], perf):
                    print_master("Creating Checkpoint: %f" % cp.min_perf_metric)
                print_master("Best Performance Metric: %f" % cp.min_perf_metric)

def update_config_frozen_conv(file_path, freeze_val:bool):
    # Load the JSON
    with open(file_path, 'r') as f:
        config = json.load(f)

    # Shortcut to the architecture section
    arch = config["NeuralNetwork"]["Architecture"]

    # Set freeze_conv_layers
    arch["freeze_conv_layers"] = freeze_val

    # Save the JSON back
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_finetuning_config(args):
    with open(args.finetuning_config, "r") as f:
        ft_config = json.load(f)
    return ft_config

def load_datasets(args, ft_config, dictionary_variables):

    # Make sure the variable config is consistent with the model config for loading datasets
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = dictionary_variables['graph_feature_names']
    var_config["graph_feature_dims"] = dictionary_variables['graph_feature_dims']
    var_config["node_feature_names"] = dictionary_variables['node_feature_names']
    var_config["node_feature_dims"] = dictionary_variables['node_feature_dims']
    # New foundation-model checkpoints expect the node feature tensor to contain
    # only atomic number. Positional information is read from `data.pos`.
    var_config["input_node_features"] = [0]

    # Load based on format
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
        fname = os.path.join(os.path.dirname(__file__), f"./dataset/{args.datasetname}.bp")
        trainset = AdiosDataset(fname, "trainset", MPI.COMM_WORLD, **opt, var_config=var_config)
        valset   = AdiosDataset(fname, "valset",   MPI.COMM_WORLD, **opt, var_config=var_config)
        testset  = AdiosDataset(fname, "testset",  MPI.COMM_WORLD, **opt, var_config=var_config)

    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(os.path.dirname(__file__), "../dataset", f"{args.datasetname}.pickle")
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
        valset   = SimplePickleDataset(basedir=basedir, label="valset",   var_config=var_config)
        testset  = SimplePickleDataset(basedir=basedir, label="testset",  var_config=var_config)

    else:
        raise NotImplementedError(f"No supported format: {args.format}")

    print("Loaded dataset.")
    info("trainset,valset,testset size: %d %d %d" % (len(trainset), len(valset), len(testset)))

    return trainset, valset, testset

def make_dataloaders(trainset, valset, testset, batch_size:int=16, ddstore=False):
    # Create dataloaders
    if ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
    (train_loader, val_loader, test_loader) = hydragnn.preprocess.create_dataloaders(trainset, valset, testset, batch_size)

    return train_loader, val_loader, test_loader

def get_ensemble(args, ft_config, train_loader, val_loader, test_loader, freeze_conv=None):
    # ---- ensemble setup ----
    ensemble_path = Path(args.pretrained_model_ensemble_path)
    model_dir_list = [os.path.join(ensemble_path, model_id) for model_id in os.listdir(ensemble_path)]
    update_config_ensemble(model_dir_list, train_loader, val_loader, test_loader)

    if (freeze_conv is not None):
        for modeldir in model_dir_list:
            update_config_frozen_conv(os.path.join(modeldir, "config.json"), freeze_val=freeze_conv)

    train_from_scratch = args.train_from_scratch or ft_config["NeuralNetwork"]["Training"].get(
        "train_from_scratch", False
    )
    model = model_ensemble(
        model_dir_list,
        fine_tune_config=ft_config,
        GFM_2024=True,
        train_from_scratch=train_from_scratch,
    )
    model = get_distributed_model(model, verbosity=2)

    return model

def setup_distributed_finetuning():
    # ---- initialize DDP / MPI ----
    comm_size, rank = setup_ddp()
    comm = MPI.COMM_WORLD
    # Return variables
    return comm_size, rank, comm

def run_finetune(dictionary_variables, args, freeze_conv:bool=None):
    """Main training/validation/test entry point.
    Accepts an argparse.Namespace-like `args`.
    """
    # ---- tracing / timers ----
    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    # ---- load fine-tuning config ----
    ft_config = load_finetuning_config(args)
    precision, param_dtype, _ = resolve_precision(
        ft_config["NeuralNetwork"]["Training"].get("precision", "fp32")
    )
    ft_config["NeuralNetwork"]["Training"]["precision"] = precision
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(param_dtype)

    try:
        # ---- default FINETUNING_LOG_DIR if not set ----
        finetuning_log_dir = os.getenv("FINETUNING_LOG_DIR")
        if not finetuning_log_dir:
            try:
                example_dir = Path(os.path.abspath(args.finetuning_config)).parent
                finetuning_log_dir = str(example_dir / "logs")
            except Exception:
                finetuning_log_dir = "./logs"
            os.environ["FINETUNING_LOG_DIR"] = finetuning_log_dir
        os.makedirs(finetuning_log_dir, exist_ok=True)

        verbosity = ft_config["Verbosity"]["level"]
        if args.batch_size is not None:
            ft_config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

        # Initialize DDP/MPI
        comm_size, rank, comm = setup_distributed_finetuning()

        # ---- logging ----
        logging.basicConfig(
            level=logging.INFO,
            format="%%(levelname)s (rank %d): %%(message)s" % (rank),
            datefmt="%H:%M:%S",
        )

        datasetname = "FineTuning" if args.datasetname is None else args.datasetname
        modelname = "FineTuning" if args.modelname is None else args.modelname
        log_name = modelname
        hydragnn.utils.print.print_utils.setup_log(log_name)
        writer = hydragnn.utils.model.get_summary_writer(log_name, path=finetuning_log_dir)

        log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

        # Get model list
        ensemble_path = Path(args.pretrained_model_ensemble_path)
        model_dir_list = [os.path.join(ensemble_path, model_id) for model_id in os.listdir(ensemble_path)]

        # Load datasets
        trainset, valset, testset = load_datasets(args, ft_config, dictionary_variables)

        # Make dataloaders
        train_loader, val_loader, test_loader = make_dataloaders(trainset, valset, testset,
                                                                batch_size=ft_config["NeuralNetwork"]["Training"]["batch_size"],
                                                                ddstore=args.ddstore)

        # Setup ensemble of models for fine-tuning
        model = get_ensemble(args, ft_config, train_loader, val_loader, test_loader, freeze_conv=freeze_conv)

        from utils.debug import print_model_sanity_check
        print_model_sanity_check(model.module.model_ens[0])
        # exit(9)

        print("Created Dataloaders")
        comm.Barrier()
        timer.stop()

        # ---- optimizers, checkpoints, train, test ----
        optimizers = [
            torch.optim.AdamW(
                member.parameters(),
                lr=ft_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"],
            )
            for member in model.module.model_ens
        ]

        # Prepare per-member checkpoints if enabled
        member_checkpoints = None
        try:
            save_ckpt = ft_config["NeuralNetwork"]["Training"].get("Checkpoint", False)
        except Exception:
            save_ckpt = False
        if save_ckpt:
            warmup = ft_config["NeuralNetwork"]["Training"].get("checkpoint_warmup", 0)
            member_checkpoints = []
            for idx, _ in enumerate(model.module.model_ens):
                # Use the original pretrained model directory name
                member_name = os.path.basename(model_dir_list[idx])
                # ensure log directory exists for this member under configured logs root
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
        )

        # Test the ensemble
        test_ensemble(model, test_loader, modelname, verbosity=2, save_results=True)

        # Optional: return something useful (e.g., final metrics)
        return True
    finally:
        torch.set_default_dtype(previous_default_dtype)
