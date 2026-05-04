#!/usr/bin/env python3
"""Evaluate a fine-tuning checkpoint on train/val/test splits."""

import sys, os, json, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import hydragnn
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config_edge_dim,
    update_config_equivariance,
)
from utils.update_model import update_model
from utils.ensemble_utils import (
    update_GFM_2024_checkpoint,
    get_distributed_model_find_unused,
)

KCAL_PER_EV = 23.0609


def main():
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    repo_root = Path(__file__).resolve().parents[2]
    example_dir = Path(__file__).resolve().parent

    # ---- fine-tuning config ---------------------------------------------------
    with open(example_dir / "finetuning_config.json") as f:
        ft_config = json.load(f)

    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = ["energy"]
    var_config["graph_feature_dims"] = [1]
    var_config["node_feature_names"] = ["atomic_number"]
    var_config["node_feature_dims"] = [1]

    # ---- datasets -------------------------------------------------------------
    basedir = str(repo_root / "dataset" / "wiggle150.pickle")
    trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
    valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=var_config)
    testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=var_config)

    batch_size = ft_config["NeuralNetwork"]["Training"]["batch_size"]
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, batch_size
    )

    # ---- rebuild model from pretrained config (without mutating the file) -----
    pretrained_dir = str(
        repo_root
        / "pretrained_model_ensemble"
        / "OneDrive_1_4-7-2026"
        / "multidataset_hpo-BEST6-fp64"
    )
    with open(os.path.join(pretrained_dir, "config.json")) as f:
        pretrained_config = json.load(f)

    arch = pretrained_config["NeuralNetwork"]["Architecture"]
    defaults = {
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
    for k, v in defaults.items():
        if k not in arch:
            arch[k] = v

    arch.update(update_config_edge_dim(arch))
    arch.update(update_config_equivariance(arch))

    training = pretrained_config["NeuralNetwork"]["Training"]
    training.setdefault("compute_grad_energy", False)
    training.setdefault("conv_checkpointing", False)

    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"], verbosity=0,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity=0)

    # load pretrained weights
    update_GFM_2024_checkpoint(
        model, os.path.basename(pretrained_dir), path=os.path.dirname(pretrained_dir)
    )

    # swap heads + freeze backbone (exactly as during training)
    model = model.module
    model = update_model(model, ft_config)
    model._freeze_conv()
    model = get_distributed_model_find_unused(model, verbosity=0)

    # ---- find best checkpoint -------------------------------------------------
    ckpt_dir = str(example_dir / "logs" / "multidataset_hpo-BEST6-fp64")
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*_epoch_*.pk"))
    if not ckpt_files:
        print("No checkpoint files found in", ckpt_dir)
        return

    def epoch_num(path):
        return int(path.rsplit("_epoch_", 1)[1].replace(".pk", ""))

    best_ckpt = max(ckpt_files, key=epoch_num)
    best_epoch = epoch_num(best_ckpt) + 1  # 0-indexed file → 1-indexed epoch

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {best_epoch} ({best_ckpt})")

    # ---- evaluate -------------------------------------------------------------
    model.eval()
    print(f"\n{'Split':>5s}  {'MAE (eV)':>10s}  {'MAE (kcal/mol)':>14s}  {'n':>4s}")
    print("-" * 42)

    for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        all_errors = []
        with torch.no_grad():
            for batch in loader:
                pred = model(batch)
                target = batch.y[:, 0]
                pred_val = pred[0].squeeze()
                errors = (pred_val - target).abs()
                all_errors.extend(errors.tolist())
        mae_eV = np.mean(all_errors)
        mae_kcal = mae_eV * KCAL_PER_EV
        print(f"{name:>5s}  {mae_eV:10.4f}  {mae_kcal:14.2f}  {len(all_errors):4d}")


if __name__ == "__main__":
    main()
