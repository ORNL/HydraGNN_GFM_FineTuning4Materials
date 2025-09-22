
import os, json, glob
import logging
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import hydragnn
from hydragnn.utils.model import print_model
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.train.train_validate_test import get_head_indices, reduce_values_ranks, gather_tensor_ranks

from hydragnn.preprocess.serialized_dataset_loader import SerializedDataLoader
from hydragnn.postprocess.postprocess import output_denormalize
from hydragnn.postprocess.visualizer import Visualizer
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm, log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.profiling_and_tracing.profile import Profiler
from hydragnn.utils.distributed import get_device, get_device_name, print_peak_memory, check_remaining
from hydragnn.preprocess.load_data import HydraDataLoader
from hydragnn.utils.model.model import Checkpoint, EarlyStopping

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    check_if_graph_size_variable,
    gather_deg,
)
from hydragnn.utils.model.model import calculate_avg_deg

from hydragnn.utils.print.print_utils import print_master

import update_model as um
import data_utils.yaml_to_config as ytc

from hydragnn.utils.input_config_parsing import update_config_edge_dim, update_config_equivariance
from hydragnn.utils.model import update_multibranch_heads

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
            config["NeuralNetwork"]["Variables_of_interest"]["input_node_features"]
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


def update_head_string(s: str) -> str:
    """
    Modify strings like:
        "module.heads_NN.0.0.weight"
    into:
        "module.heads_NN.0.branch-0.0.weight"
    """
    parts = s.split(".")
    # Insert "branch-" before the second index
    if len(parts) > 3:
        parts[3] = f"branch-0.{parts[3]}"
        return ".".join([parts[0], parts[1], parts[2], parts[3]] + parts[4:])
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
    if len(parts) > 2:
        parts[2] = f"branch-0.{parts[2]}"
        return ".".join([parts[0], parts[1], parts[2], parts[3]])
    return s


def update_GFM_2024_checkpoint(
    model, model_name, path="./logs/", optimizer=None, use_deepspeed=False
):
    """Load both model and optimizer state from a single checkpoint file, renaming head keys."""
    path_name = os.path.join(path, model_name, model_name + ".pk")
    map_location = {"cuda:%d" % 0: get_device_name()}
    print_master("Load existing model:", path_name)

    checkpoint = torch.load(path_name, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]

    # Build a renamed copy so we don't modify while iterating
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

    # Load into model
    model.load_state_dict(state_dict)

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
            with open(input_filename, "r") as f:
                config = json.load(f)
            model = hydragnn.models.create_model_config(
                config=config["NeuralNetwork"],
                verbosity=verbosity,
            )
            model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
            # Print details of neural network architecture
            # print("Loading model %d, %s"%(imodel, modeldir))
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
                model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

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
        total_tasks_loss = torch.zeros(ntasks)
        for k, model in enumerate(self.model_ens):
            head_index = get_head_indices(model, data)
            pred = model(data)
            # Compute loss for the k-th model
            loss, tasks_loss = model.module.loss(pred, data.y.float(), head_index)

            model.zero_grad()  # Zero out gradients specific to this model
            loss.backward()  # Backpropagate for the current model

            # Update task-wise losses
            total_tasks_loss = torch.mean(total_tasks_loss + torch.Tensor(tasks_loss), dim=0)
        return total_tasks_loss  # Optionally return task-wise accumulated losses
    
    def val_loss(self, data): 
        ntasks = data.y.shape[1]
        total_tasks_loss = torch.zeros(ntasks)
        for k, model in enumerate(self.model_ens):
            head_index = get_head_indices(model, data)
            pred = model(data)
            # Compute loss for the k-th model
            loss, tasks_loss = model.module.loss(pred, data.y.float(), head_index)

            model.zero_grad()  # Zero out gradients specific to this model
            

            # Update task-wise losses
            total_tasks_loss = torch.mean(total_tasks_loss + torch.Tensor(tasks_loss), dim=0)
        return total_tasks_loss  # Optionally return task-wise accumulated losses

    def __len__(self):
        return self.model_size


def test_ensemble(model_ens, loader, dataset_name, verbosity, num_samples=None):
    n_ens=len(model_ens.module)
    num_heads=model_ens.module.num_heads

    num_samples_total = 0
    device = next(model_ens.parameters()).device

    true_values = [[] for _ in range(num_heads)]
    predicted_values = [[[] for _ in range(num_heads)] for _ in range(n_ens)]
   
    for data in iterate_tqdm(loader, verbosity):
        data=data.to(device)
        head_index = get_head_indices(model_ens.module.model_ens[0], data)
        ###########################
        pred_ens = model_ens(data)
        ###########################
        ytrue = data.y
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
       # print("For head %d"%ihead)
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
        # print(head_pred_ens.size(), true_values[ihead].size())
    
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
    outputs_df.to_csv(f"dataset_predictions/{dataset_name}_predictions.csv", index=False)

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

def train_ensemble(model_ensemble, train_loader, val_loader, num_epochs, optimizers, device="cpu"):
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
                 range(num_epochs), verbosity_level=2, desc="Train Ensemble", total=num_epochs
        ):
        model_ensemble.train()  # Set ensemble to training mode
        epoch_loss = 0  # Track cumulative loss for the epoch
        for batch in train_loader:
            # Forward pass and backpropagation
            total_tasks_loss = torch.atleast_1d(model_ensemble.module.loss_and_backprop(batch.to(get_device())))

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
            total_tasks_loss = torch.atleast_1d(model_ensemble.module.val_loss(batch.to(get_device())))

            #accumulate validation loss for logging 
            val_loss += total_tasks_loss[0]

        mean_val_loss_epoch = val_loss/len(val_loader)
    
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_train_loss_epoch:.4f}, Val Loss: {mean_val_loss_epoch:.4f}")


