#!/usr/bin/env python3
"""Quick parameter count for the different model configurations."""
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "HydraGNN"))
sys.path.insert(0, str(REPO))

import hydragnn
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config_edge_dim, update_config_equivariance,
)
from utils.update_model import update_model

D = str(REPO / "pretrained_model_ensemble/OneDrive_1_4-7-2026/multidataset_hpo-BEST6-fp64")
with open(os.path.join(D, "config.json")) as f:
    pcfg = json.load(f)

arch = pcfg["NeuralNetwork"]["Architecture"]
for k, v in {
    "global_attn_engine": None, "global_attn_type": None, "global_attn_heads": 0,
    "pe_dim": 0, "pna_deg": None, "freeze_conv_layers": False,
    "initial_bias": None, "activation_function": "relu", "SyncBatchNorm": False,
    "radius": None, "radial_type": None, "distance_transform": None,
    "num_gaussians": None, "num_filters": None, "envelope_exponent": None,
    "num_after_skip": None, "num_before_skip": None, "basis_emb_size": None,
    "int_emb_size": None, "out_emb_size": None, "num_radial": None,
    "num_spherical": None, "correlation": None, "max_ell": None,
    "node_max_ell": None, "avg_num_neighbors": None,
}.items():
    arch.setdefault(k, v)
arch.update(update_config_edge_dim(arch))
arch.update(update_config_equivariance(arch))
pcfg["NeuralNetwork"]["Training"].setdefault("compute_grad_energy", False)
pcfg["NeuralNetwork"]["Training"].setdefault("conv_checkpointing", False)

def count(model):
    bb = sum(p.numel() for n, p in model.named_parameters()
             if "heads_NN" not in n and "graph_shared" not in n)
    sh = sum(p.numel() for n, p in model.named_parameters()
             if "graph_shared" in n)
    hd = sum(p.numel() for n, p in model.named_parameters()
             if "heads_NN" in n)
    return bb, sh, hd

# Config 1: scratch/frozen/unfrozen
with open(str(REPO / "examples/md17/finetuning_config_mlip.json")) as f:
    ft1 = json.load(f)
m1 = hydragnn.models.create_model_config(config=pcfg["NeuralNetwork"], verbosity=0)
m1 = update_model(m1, ft1)
bb1, sh1, hd1 = count(m1)
print(f"scratch/frozen/unfrozen (dim_pretrained=128, headlayers=[128,64]):")
print(f"  backbone={bb1:,}  shared={sh1:,}  head={hd1:,}  total={bb1+sh1+hd1:,}")
print(f"  trainable(scratch)  = {bb1+sh1+hd1:,}")
print(f"  trainable(frozen)   = {sh1+hd1:,}")
print(f"  trainable(unfrozen) = {bb1+sh1+hd1:,}")

# Config 2: ANI1x recycled
with open(str(REPO / "examples/md17/finetuning_config_mlip_ani1x.json")) as f:
    ft2 = json.load(f)
m2 = hydragnn.models.create_model_config(config=pcfg["NeuralNetwork"], verbosity=0)
m2 = update_model(m2, ft2)
bb2, sh2, hd2 = count(m2)
print(f"\nani1x_recycled (dim_pretrained=50, headlayers=[776,776]):")
print(f"  backbone={bb2:,}  shared={sh2:,}  head={hd2:,}  total={bb2+sh2+hd2:,}")

# Training samples
print(f"\nTraining samples: 350")
print(f"Params-to-samples ratio (scratch): {(bb1+sh1+hd1)/350:.0f}:1")
print(f"Params-to-samples ratio (ani1x):   {(bb2+sh2+hd2)/350:.0f}:1")
