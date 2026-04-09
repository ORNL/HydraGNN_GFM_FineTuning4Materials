#!/usr/bin/env python3
"""Unified benchmark: frozen fine-tuning, unfrozen fine-tuning, and scratch training
across BF16, FP32, and FP64 precisions on the Wiggle150 dataset.

Generates a validation-MAE-vs-epoch plot for FP64 runs.
"""

import sys, os, json, copy
from pathlib import Path

# Ensure project roots are on the path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HydraGNN"))
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

import hydragnn
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config_edge_dim,
    update_config_equivariance,
)
from hydragnn.train.train_validate_test import (
    get_head_indices,
    resolve_precision,
    move_batch_to_device,
    get_autocast_and_scaler,
)

from utils.update_model import update_model as _update_model
from utils.ensemble_utils import (
    update_GFM_2024_checkpoint,
    get_distributed_model_find_unused,
    _force_dataset_name_2d,
)

# Default graph_attr: charge=0, spin=1
_DEFAULT_GRAPH_ATTR = torch.tensor([0.0, 1.0])


def _ensure_graph_attr(batch):
    """Inject default graph_attr (charge=0, spin=1) if missing."""
    if not hasattr(batch, "graph_attr") or batch.graph_attr is None:
        num_graphs = int(batch.batch.max().item() + 1) if hasattr(batch, "batch") else 1
        batch.graph_attr = _DEFAULT_GRAPH_ATTR.unsqueeze(0).expand(num_graphs, -1).clone()
    return batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRETRAINED_DIR = str(
    REPO_ROOT
    / "pretrained_model_ensemble"
    / "OneDrive_1_4-7-2026"
    / "multidataset_hpo-BEST6-fp64"
)
FT_CONFIG_PATH = str(REPO_ROOT / "examples" / "wiggle150" / "finetuning_config.json")
DATASET_DIR = str(REPO_ROOT / "dataset" / "wiggle150.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "wiggle150" / "benchmark_results")

NUM_EPOCHS = 500
BATCH_SIZE = 32

# Precision is resolved dynamically via HydraGNN's resolve_precision()

# Architecture defaults that may be missing from the pretrained config
ARCH_DEFAULTS = {
    "global_attn_engine": None, "global_attn_type": None, "global_attn_heads": 0,
    "pe_dim": 0, "pna_deg": None, "freeze_conv_layers": False,
    "initial_bias": None, "activation_function": "relu", "SyncBatchNorm": False,
    "radius": None, "radial_type": None, "distance_transform": None,
    "num_gaussians": None, "num_filters": None, "envelope_exponent": None,
    "num_after_skip": None, "num_before_skip": None, "basis_emb_size": None,
    "int_emb_size": None, "out_emb_size": None, "num_radial": None,
    "num_spherical": None, "correlation": None, "max_ell": None,
    "node_max_ell": None, "avg_num_neighbors": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_ft_config():
    with open(FT_CONFIG_PATH) as f:
        return json.load(f)


def load_pretrained_config():
    with open(os.path.join(PRETRAINED_DIR, "config.json")) as f:
        cfg = json.load(f)
    arch = cfg["NeuralNetwork"]["Architecture"]
    for k, v in ARCH_DEFAULTS.items():
        arch.setdefault(k, v)
    arch.update(update_config_edge_dim(arch))
    arch.update(update_config_equivariance(arch))
    training = cfg["NeuralNetwork"]["Training"]
    training.setdefault("compute_grad_energy", False)
    training.setdefault("conv_checkpointing", False)
    return cfg


def make_dataloaders(ft_config):
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = ["energy"]
    var_config["graph_feature_dims"] = [1]
    var_config["node_feature_names"] = ["atomic_number"]
    var_config["node_feature_dims"] = [1]
    var_config["input_node_features"] = [0]

    trainset = SimplePickleDataset(basedir=DATASET_DIR, label="trainset", var_config=var_config)
    valset = SimplePickleDataset(basedir=DATASET_DIR, label="valset", var_config=var_config)
    testset = SimplePickleDataset(basedir=DATASET_DIR, label="testset", var_config=var_config)
    return hydragnn.preprocess.create_dataloaders(trainset, valset, testset, BATCH_SIZE)


def build_model_from_pretrained(pretrained_config, ft_config, freeze):
    """Build a model from pretrained config, load pretrained weights, swap heads."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity=0)

    # Load pretrained weights
    update_GFM_2024_checkpoint(
        model,
        os.path.basename(PRETRAINED_DIR),
        path=os.path.dirname(PRETRAINED_DIR),
    )

    # Unwrap, swap heads, optionally freeze, re-wrap
    model = model.module
    model = _update_model(model, ft_config)
    if freeze:
        model._freeze_conv()
    return model


def build_scratch_model(pretrained_config, ft_config):
    """Build a model from architecture template with random weights."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = _update_model(model, ft_config)
    return model


KCAL_PER_EV = 23.0609


def train_loop(model, train_loader, val_loader, num_epochs, optimizer, precision):
    """Train for num_epochs, return dict with per-epoch val MAE and RMSE (in eV).

    Uses HydraGNN's precision infrastructure:
    - resolve_precision() for param_dtype
    - move_batch_to_device() for dtype-aware batch casting
    - get_autocast_and_scaler() for bf16 autocast
    """
    prec, param_dtype, _ = resolve_precision(precision)
    autocast_ctx, scaler = get_autocast_and_scaler(prec)
    mae_history = []
    rmse_history = []

    for epoch in range(num_epochs):
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)

        # ----- Train -----
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = _force_dataset_name_2d(batch)
            batch = _ensure_graph_attr(batch)
            batch = move_batch_to_device(batch, param_dtype)

            head_index = get_head_indices(model, batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred = model(batch)
                loss_fn_owner = model.module if hasattr(model, "module") else model
                loss, tasks_loss = loss_fn_owner.loss(pred, batch.y, head_index)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += float(torch.atleast_1d(tasks_loss)[0].detach().cpu())

        mean_train = epoch_loss / len(train_loader)

        # ----- Validate -----
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _force_dataset_name_2d(batch)
                batch = _ensure_graph_attr(batch)
                batch = move_batch_to_device(batch, param_dtype)

                head_index = get_head_indices(model, batch)
                with autocast_ctx:
                    pred = model(batch)
                # Collect predictions and targets for head 0
                head_pred = pred[0].detach().cpu().float().view(-1)
                head_true = batch.y[head_index[0]].detach().cpu().float().view(-1)
                all_pred.append(head_pred)
                all_true.append(head_true)

        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)
        errors = all_pred - all_true
        val_mae = float(errors.abs().mean())
        val_rmse = float((errors**2).mean().sqrt())
        mae_history.append(val_mae)
        rmse_history.append(val_rmse)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1:4d}/{num_epochs}"
                f"  Train MAE: {mean_train:.4f}"
                f"  Val MAE: {val_mae:.4f}  Val RMSE: {val_rmse:.4f}"
            )

    return {"mae": mae_history, "rmse": rmse_history}


def run_experiment(strategy, precision_name, ft_config, pretrained_config, train_loader, val_loader):
    """Run a single (strategy, precision) experiment. Returns val_history list."""
    prec, param_dtype, _ = resolve_precision(precision_name)
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}  |  Precision: {prec}  (param_dtype={param_dtype})")
    print(f"{'='*60}")

    freeze = strategy == "frozen"

    if strategy in ("frozen", "unfrozen"):
        model = build_model_from_pretrained(pretrained_config, ft_config, freeze=freeze)
    else:  # scratch
        model = build_scratch_model(pretrained_config, ft_config)

    # Cast model parameters to param_dtype (FP32 for bf16, FP32 for fp32, FP64 for fp64)
    model = model.to(dtype=param_dtype)
    model = get_distributed_model_find_unused(model, verbosity=0)

    # Optimizer
    lr = ft_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    wd = ft_config["NeuralNetwork"]["Training"]["Optimizer"].get("weight_decay", 0.0)

    # Use lower lr for unfrozen to protect backbone
    if strategy == "unfrozen":
        lr = lr * 0.1

    if freeze:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    history = train_loop(model, train_loader, val_loader, NUM_EPOCHS, optimizer, prec)

    best_mae = min(history["mae"])
    best_mae_epoch = history["mae"].index(best_mae) + 1
    best_rmse = min(history["rmse"])
    best_rmse_epoch = history["rmse"].index(best_rmse) + 1
    print(
        f"  Best Val MAE:  {best_mae:.4f} eV ({best_mae * KCAL_PER_EV:.2f} kcal/mol) at epoch {best_mae_epoch}\n"
        f"  Best Val RMSE: {best_rmse:.4f} eV ({best_rmse * KCAL_PER_EV:.2f} kcal/mol) at epoch {best_rmse_epoch}"
    )

    return history


def plot_fp64_curves(results, output_dir):
    """Plot validation MAE and RMSE vs epoch for the three FP64 strategies."""
    labels = {
        "frozen": "Fine-tuning (frozen backbone)",
        "unfrozen": "Fine-tuning (unfrozen backbone)",
        "scratch": "Training from scratch",
    }
    colors = {
        "frozen": "#1f77b4",
        "unfrozen": "#ff7f0e",
        "scratch": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in zip(
        axes, ("mae", "rmse"), ("Validation MAE (eV)", "Validation RMSE (eV)")
    ):
        for strategy in ("frozen", "unfrozen", "scratch"):
            key = f"{strategy}_fp64"
            if key not in results:
                continue
            hist = results[key][metric]
            epochs = list(range(1, len(hist) + 1))
            ax.plot(epochs, hist, label=labels[strategy], color=colors[strategy], linewidth=1.2)

        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "fp64_validation_mae_rmse.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    world_size, world_rank = setup_ddp()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ft_config_template = load_ft_config()
    pretrained_config = load_pretrained_config()

    # Build dataloaders once (shared across all runs)
    train_loader, val_loader, test_loader = make_dataloaders(copy.deepcopy(ft_config_template))

    strategies = ["frozen", "unfrozen", "scratch"]
    precisions = ["bf16", "fp32", "fp64"]

    results = {}  # key: "{strategy}_{dtype}" -> val_history list

    for prec_name in precisions:
        for strategy in strategies:
            ft_config = copy.deepcopy(ft_config_template)
            key = f"{strategy}_{prec_name}"
            val_history = run_experiment(
                strategy, prec_name, ft_config, pretrained_config,
                train_loader, val_loader,
            )
            results[key] = val_history

    # Save all results to JSON
    summary = {}
    for key, hist in results.items():
        mae_list = hist["mae"]
        rmse_list = hist["rmse"]
        best_mae = min(mae_list)
        best_rmse = min(rmse_list)
        summary[key] = {
            "best_val_mae_eV": best_mae,
            "best_mae_epoch": mae_list.index(best_mae) + 1,
            "best_val_mae_kcal_mol": best_mae * KCAL_PER_EV,
            "best_val_rmse_eV": best_rmse,
            "best_rmse_epoch": rmse_list.index(best_rmse) + 1,
            "best_val_rmse_kcal_mol": best_rmse * KCAL_PER_EV,
            "final_val_mae_eV": mae_list[-1],
            "final_val_rmse_eV": rmse_list[-1],
        }
    summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save per-epoch histories
    history_path = os.path.join(OUTPUT_DIR, "val_histories.json")
    with open(history_path, "w") as f:
        json.dump(results, f)
    print(f"Per-epoch histories saved to {history_path}")

    # Plot FP64 curves
    plot_fp64_curves(results, OUTPUT_DIR)

    # Print final summary table
    print(f"\n{'Strategy':<22s}  {'Prec':<5s}  {'MAE(eV)':>8s}  {'MAE(kcal)':>10s}  {'Ep':>4s}  {'RMSE(eV)':>9s}  {'RMSE(kcal)':>11s}  {'Ep':>4s}")
    print("-" * 90)
    for prec_name in precisions:
        for strategy in strategies:
            key = f"{strategy}_{prec_name}"
            s = summary[key]
            print(
                f"{strategy:<22s}  {prec_name:<5s}"
                f"  {s['best_val_mae_eV']:8.4f}  {s['best_val_mae_kcal_mol']:10.2f}  {s['best_mae_epoch']:4d}"
                f"  {s['best_val_rmse_eV']:9.4f}  {s['best_val_rmse_kcal_mol']:11.2f}  {s['best_rmse_epoch']:4d}"
            )


if __name__ == "__main__":
    main()
