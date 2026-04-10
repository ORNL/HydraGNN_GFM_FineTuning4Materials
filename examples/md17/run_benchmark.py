#!/usr/bin/env python3
"""MD17 MLIP benchmark: fine-tuning (unfrozen) vs scratch training with
energy + forces (interatomic potential mode).

The pretrained GFM already supports graph-level energy heads with autograd
forces via HydraGNN's EnhancedModelWrapper.  We keep a single graph-level
head and train with compute_grad_energy=True so the loss includes:
  energy_weight * MSE(E_pred, E_true) + force_weight * MSE(F_pred, F_true)
"""

import sys, os, json, copy
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRETRAINED_DIR = str(
    REPO_ROOT / "pretrained_model_ensemble" / "OneDrive_1_4-7-2026"
    / "multidataset_hpo-BEST6-fp64"
)
FT_CONFIG_PATH = str(REPO_ROOT / "examples" / "md17" / "finetuning_config_mlip.json")
FT_CONFIG_ANI1X_PATH = str(REPO_ROOT / "examples" / "md17" / "finetuning_config_mlip_ani1x.json")
DATASET_DIR = str(REPO_ROOT / "dataset" / "md17_mlip.pickle")
OUTPUT_DIR = str(REPO_ROOT / "examples" / "md17" / "benchmark_results")

NUM_EPOCHS = 100
WARMUP_EPOCHS = 20
BACKBONE_LR_MULT = 0.1  # fine-tuning: backbone lr = base_lr * this
BATCH_SIZE = 32

_DEFAULT_GRAPH_ATTR = torch.tensor([0.0, 1.0])

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

# Units: Dataset is stored in eV (energy) and eV/Å (forces) after preprocessing.
# Conversion factor for reporting in kcal/mol alongside eV.
KCAL_PER_EV = 23.0609


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_graph_attr(batch):
    if not hasattr(batch, "graph_attr") or batch.graph_attr is None:
        ng = int(batch.batch.max().item() + 1) if hasattr(batch, "batch") else 1
        batch.graph_attr = _DEFAULT_GRAPH_ATTR.unsqueeze(0).expand(ng, -1).clone()
    return batch


def load_ft_config():
    with open(FT_CONFIG_PATH) as f:
        return json.load(f)


def load_ft_config_ani1x():
    with open(FT_CONFIG_ANI1X_PATH) as f:
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


def build_model_from_pretrained(pretrained_config, ft_config, freeze=False):
    """Build model from pretrained backbone, swap to single MLIP head."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity=0)

    update_GFM_2024_checkpoint(
        model,
        os.path.basename(PRETRAINED_DIR),
        path=os.path.dirname(PRETRAINED_DIR),
    )

    # Unwrap from DDP, swap heads, optionally freeze
    model = model.module
    model = _update_model(model, ft_config)
    if freeze:
        model._freeze_conv()

    # Propagate MLIP loss weights from ft_config into the wrapper
    arch = ft_config["NeuralNetwork"]["Architecture"]
    if hasattr(model, "energy_weight"):
        model.energy_weight = arch.get("energy_weight", 1.0)
        model.energy_peratom_weight = arch.get("energy_peratom_weight", 0.0)
        model.force_weight = arch.get("force_weight", 100.0)

    return model


def build_scratch_model(pretrained_config, ft_config):
    """Build model with random weights using the same architecture."""
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = _update_model(model, ft_config)

    arch = ft_config["NeuralNetwork"]["Architecture"]
    if hasattr(model, "energy_weight"):
        model.energy_weight = arch.get("energy_weight", 1.0)
        model.energy_peratom_weight = arch.get("energy_peratom_weight", 0.0)
        model.force_weight = arch.get("force_weight", 100.0)

    return model


ANI1X_BRANCH = 1  # branch-1 = ANI1x in the pretrained 16-head model


def build_model_with_recycled_head(pretrained_config, ft_config, source_branch=ANI1X_BRANCH, freeze=False):
    """Build model from pretrained backbone, recycling the ANI1x head (branch-1).

    Instead of creating a random head, this copies the pretrained shared layers
    and head layers from the source branch into branch-0 of the fine-tuning model.
    The ft_config must have dim_pretrained/dim_headlayers matching the original head.
    """
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity=0)

    update_GFM_2024_checkpoint(
        model,
        os.path.basename(PRETRAINED_DIR),
        path=os.path.dirname(PRETRAINED_DIR),
    )

    # Unwrap from DDP
    model = model.module

    # Save the source branch module state_dicts before update_model replaces heads
    src_tag = f"branch-{source_branch}"

    # graph_shared is a ModuleDict keyed by branch type
    saved_shared_state = model.graph_shared[src_tag].state_dict()

    # heads_NN is a ModuleList of ModuleDicts; find the one with our source branch
    saved_head_state = None
    for head_dict in model.heads_NN:
        if src_tag in head_dict:
            saved_head_state = head_dict[src_tag].state_dict()
            break

    n_shared = sum(1 for _ in saved_shared_state)
    n_head = sum(1 for _ in saved_head_state) if saved_head_state else 0
    print(f"  Recycling pretrained {src_tag} (ANI1x) → branch-0")
    print(f"    Saved {n_shared} shared params, {n_head} head params")

    # Replace heads with matching architecture (dim_pretrained=50, dim_headlayers=[776,776])
    model = _update_model(model, ft_config)

    # Copy saved weights into the new branch-0 modules directly
    with torch.no_grad():
        model.graph_shared["branch-0"].load_state_dict(saved_shared_state)
        if saved_head_state is not None:
            model.heads_NN[0]["branch-0"].load_state_dict(saved_head_state)

    if freeze:
        model._freeze_conv()

    # Propagate MLIP loss weights
    arch = ft_config["NeuralNetwork"]["Architecture"]
    if hasattr(model, "energy_weight"):
        model.energy_weight = arch.get("energy_weight", 1.0)
        model.energy_peratom_weight = arch.get("energy_peratom_weight", 0.0)
        model.force_weight = arch.get("force_weight", 100.0)

    return model


# ---------------------------------------------------------------------------
# Optimizer / Scheduler helpers
# ---------------------------------------------------------------------------
def make_param_groups(model, base_lr, strategy):
    """Create optimizer param groups with optional differential lr."""
    use_differential = False  # uniform LR for all strategies

    if strategy == "frozen":
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr}]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.replace("module.", "", 1) if name.startswith("module.") else name
        if "heads_NN" in clean or "graph_shared" in clean:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if use_differential:
        bb_lr = base_lr * BACKBONE_LR_MULT
        print(f"    Differential lr: backbone={bb_lr:.2e}, heads={base_lr:.2e}")
        return [
            {"params": backbone_params, "lr": bb_lr},
            {"params": head_params, "lr": base_lr},
        ]
    else:
        return [{"params": backbone_params + head_params, "lr": base_lr}]


def make_scheduler(optimizer, num_epochs, warmup_epochs):
    """Linear warmup then cosine annealing to near-zero."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )
    constant = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=num_epochs - warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, constant], milestones=[warmup_epochs],
    )


# ---------------------------------------------------------------------------
# Training loop with energy + force loss
# ---------------------------------------------------------------------------
def train_loop(model, train_loader, val_loader, num_epochs, optimizer, precision,
               scheduler=None):
    """Train with energy_force_loss, return per-epoch metrics dict.

    Returns dict with keys:
      energy_mae, force_mae  (validation, per epoch)
    """
    prec, param_dtype, _ = resolve_precision(precision)
    autocast_ctx, scaler = get_autocast_and_scaler(prec)
    device = get_device()

    energy_mae_hist = []
    force_mae_hist = []

    for epoch in range(num_epochs):
        os.environ["HYDRAGNN_EPOCH"] = str(epoch)

        # ---------- Train ----------
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = _force_dataset_name_2d(batch)
            batch = _ensure_graph_attr(batch)
            batch = move_batch_to_device(batch, param_dtype)
            # Enable grad on positions for force computation
            batch.pos.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                pred = model(batch)
                loss_owner = model.module if hasattr(model, "module") else model
                loss, tasks_loss = loss_owner.energy_force_loss(pred, batch)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        mean_train_loss = epoch_loss / len(train_loader)

        # ---------- Validate ----------
        model.eval()
        all_energy_pred = []
        all_energy_true = []
        all_force_pred = []
        all_force_true = []

        for batch in val_loader:
            batch = _force_dataset_name_2d(batch)
            batch = _ensure_graph_attr(batch)
            batch = move_batch_to_device(batch, param_dtype)
            batch.pos.requires_grad_(True)

            with torch.enable_grad():
                pred = model(batch)

                # Extract graph energy
                loss_owner = model.module if hasattr(model, "module") else model
                if loss_owner.head_type[0] == "graph":
                    if isinstance(pred, (list, tuple)):
                        e_pred = pred[0].squeeze().float()
                    else:
                        e_pred = pred.squeeze().float()
                else:
                    import torch_scatter
                    e_pred = torch_scatter.scatter_add(
                        pred[0], batch.batch, dim=0
                    ).squeeze().float()

                e_true = batch.energy.squeeze().float()

                # Compute forces via autograd
                f_pred = torch.autograd.grad(
                    e_pred, batch.pos,
                    grad_outputs=torch.ones_like(e_pred),
                    retain_graph=False, create_graph=False,
                )[0].float()
                f_pred = -f_pred

            f_true = batch.forces.float()

            all_energy_pred.append(e_pred.detach().cpu())
            all_energy_true.append(e_true.detach().cpu())
            all_force_pred.append(f_pred.detach().cpu())
            all_force_true.append(f_true.detach().cpu())

        all_energy_pred = torch.cat(all_energy_pred)
        all_energy_true = torch.cat(all_energy_true)
        all_force_pred = torch.cat(all_force_pred)
        all_force_true = torch.cat(all_force_true)

        energy_mae = float((all_energy_pred - all_energy_true).abs().mean())
        force_mae = float((all_force_pred - all_force_true).abs().mean())

        energy_mae_hist.append(energy_mae)
        force_mae_hist.append(force_mae)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_parts = " ".join(f"lr{i}={pg['lr']:.2e}" for i, pg in enumerate(optimizer.param_groups))
            print(
                f"    Epoch {epoch+1:4d}/{num_epochs}"
                f"  Loss: {mean_train_loss:.4f}"
                f"  Val E-MAE: {energy_mae:.4f} eV ({energy_mae * KCAL_PER_EV:.4f} kcal/mol)"
                f"  Val F-MAE: {force_mae:.4f} eV/Å ({force_mae * KCAL_PER_EV:.4f} kcal/(mol·Å))"
                f"  [{lr_parts}]"
            )

    return {"energy_mae": energy_mae_hist, "force_mae": force_mae_hist}


# ---------------------------------------------------------------------------
# Run single experiment
# ---------------------------------------------------------------------------
def run_experiment(strategy, ft_config, pretrained_config, train_loader, val_loader):
    precision = "fp64"
    prec, param_dtype, _ = resolve_precision(precision)

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}  |  MLIP (energy + forces)")
    print(f"{'='*60}")

    if strategy in ("frozen", "unfrozen"):
        freeze = (strategy == "frozen")
        model = build_model_from_pretrained(pretrained_config, ft_config, freeze=freeze)
    elif strategy == "ani1x_recycled":
        model = build_model_with_recycled_head(pretrained_config, ft_config, freeze=False)
    else:
        model = build_scratch_model(pretrained_config, ft_config)

    model = model.to(dtype=param_dtype)
    model = get_distributed_model_find_unused(model, verbosity=0)

    lr = ft_config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    wd = ft_config["NeuralNetwork"]["Training"]["Optimizer"].get("weight_decay", 0.0)

    param_groups = make_param_groups(model, lr, strategy)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    scheduler = make_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)

    print(f"    Scheduler: warmup {WARMUP_EPOCHS} epochs → constant lr for {NUM_EPOCHS - WARMUP_EPOCHS} epochs")

    history = train_loop(model, train_loader, val_loader, NUM_EPOCHS, optimizer,
                         precision, scheduler=scheduler)

    best_e = min(history["energy_mae"])
    best_e_ep = history["energy_mae"].index(best_e) + 1
    best_f = min(history["force_mae"])
    best_f_ep = history["force_mae"].index(best_f) + 1
    print(f"  Best Val Energy MAE: {best_e:.4f} eV  ({best_e * KCAL_PER_EV:.4f} kcal/mol) at epoch {best_e_ep}")
    print(f"  Best Val Force  MAE: {best_f:.4f} eV/Å ({best_f * KCAL_PER_EV:.4f} kcal/(mol·Å)) at epoch {best_f_ep}")

    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(results, output_dir):
    labels = {
        "frozen": "Fine-tuning (frozen)",
        "unfrozen": "Fine-tuning (unfrozen)",
        "scratch": "From scratch",
        "ani1x_recycled": "ANI1x head recycled",
    }
    colors = {"frozen": "#1f77b4", "unfrozen": "#ff7f0e", "scratch": "#2ca02c", "ani1x_recycled": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in zip(
        axes,
        ("energy_mae", "force_mae"),
        ("Energy MAE (eV)", "Force MAE (eV/Å)"),
    ):
        for strategy in ("frozen", "unfrozen", "scratch", "ani1x_recycled"):
            if strategy not in results:
                continue
            hist = results[strategy][metric]
            epochs = list(range(1, len(hist) + 1))
            ax.plot(epochs, hist, label=labels[strategy], color=colors[strategy], lw=1.2)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MD17 Uracil MLIP — Energy + Force Training", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "md17_mlip_validation.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    world_size, world_rank = setup_ddp()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use ANI1x config for ALL strategies so head architecture is identical
    ft_config_template = load_ft_config_ani1x()
    pretrained_config = load_pretrained_config()

    train_loader, val_loader, test_loader = make_dataloaders(
        copy.deepcopy(ft_config_template)
    )

    strategies = ["scratch", "unfrozen", "ani1x_recycled", "frozen"]
    results = {}

    for strategy in strategies:
        ft_config = copy.deepcopy(ft_config_template)
        results[strategy] = run_experiment(
            strategy, ft_config, pretrained_config, train_loader, val_loader
        )

    # Save summary
    summary = {}
    for strategy, hist in results.items():
        best_e = min(hist["energy_mae"])
        best_f = min(hist["force_mae"])
        summary[strategy] = {
            "best_energy_mae_eV": best_e,
            "best_energy_mae_kcal_mol": best_e * KCAL_PER_EV,
            "best_energy_epoch": hist["energy_mae"].index(best_e) + 1,
            "best_force_mae_eV_A": best_f,
            "best_force_mae_kcal_mol_A": best_f * KCAL_PER_EV,
            "best_force_epoch": hist["force_mae"].index(best_f) + 1,
        }

    summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    history_path = os.path.join(OUTPUT_DIR, "val_histories.json")
    with open(history_path, "w") as f:
        json.dump(results, f)
    print(f"Histories saved to {history_path}")

    plot_curves(results, OUTPUT_DIR)

    # Print table
    print(f"\n{'Strategy':<18s}  {'E-MAE (eV)':>10s}  {'E-MAE (kcal/mol)':>17s}  {'Ep':>4s}  {'F-MAE (eV/Å)':>13s}  {'F-MAE (kcal/(mol·Å))':>21s}  {'Ep':>4s}")
    print("-" * 100)
    for strategy in strategies:
        s = summary[strategy]
        print(
            f"{strategy:<18s}"
            f"  {s['best_energy_mae_eV']:10.4f}"
            f"  {s['best_energy_mae_kcal_mol']:17.4f}"
            f"  {s['best_energy_epoch']:4d}"
            f"  {s['best_force_mae_eV_A']:13.4f}"
            f"  {s['best_force_mae_kcal_mol_A']:21.4f}"
            f"  {s['best_force_epoch']:4d}"
        )


if __name__ == "__main__":
    main()
