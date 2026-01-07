#!/usr/bin/env python3
from pathlib import Path
import os
import json

from utils.ensemble_utils import build_arg_parser, run_finetune

# (radius [Å], max_neighbours)
MS25_CUTOFFS = {
    "MgO-2x2":  (6.0, 64),
    "MgO-4x4":  (6.0, 64),
    "H2O-64":   (6.5, 48),
    "H2O-192":  (6.5, 48),
    "CHA":      (5.0, 64),
    "HEA":      (5.5, 64),
    "Reaction": (6.0, 64),
    "Zr-O":     (6.0, 64),
}

def infer_system(modelname: str) -> str:
    # allows modelname like "HEA_seed0" => "HEA"
    return modelname.split("_")[0]

if __name__ == "__main__":
    parser = build_arg_parser()
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Explicit MS25 system name. Overrides modelname parsing.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent          # .../examples/ms25
    repo = here.parent.parent                       # .../HydraGNN_GFM_FineTuning4Materials

    modelname = "MS25" if args.modelname is None else args.modelname
    system = args.system or infer_system(modelname)

    if system not in MS25_CUTOFFS:
        raise ValueError(f"Unknown system '{system}'. Valid: {list(MS25_CUTOFFS.keys())}")

    radius, max_nbrs = MS25_CUTOFFS[system]

    if not getattr(args, "pretrained_model_ensemble_path", None):
        args.pretrained_model_ensemble_path = str((repo / "pretrained_model_ensemble").resolve())

    base_cfg_path = (here / "finetuning_config.json").resolve()
    if getattr(args, "finetuning_config", None):
        base_cfg_path = Path(args.finetuning_config).resolve()

    # Patch config to temp file 
    with open(base_cfg_path, "r") as f:
        cfg = json.load(f)

    cfg.setdefault("NeuralNetwork", {}).setdefault("Architecture", {})
    cfg["NeuralNetwork"]["Architecture"]["radius"] = float(radius)
    cfg["NeuralNetwork"]["Architecture"]["max_neighbours"] = int(max_nbrs)
    cfg["NeuralNetwork"]["Architecture"]["periodic_boundary_conditions"] = True

    patched_cfg_path = here / f"finetuning_config_patched_{system}.json"
    with open(patched_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    args.finetuning_config = str(patched_cfg_path.resolve())

    # Feature schema override 
    dictionary_variables = {
        "graph_feature_names": ["energy"],
        "graph_feature_dims": [1],
        "node_feature_names": ["atomic_number", "cartesian_coordinates"],
        "node_feature_dims": [1, 3],
    }

    # Place all logs/checkpoints under the example folder
    os.environ["FINETUNING_LOG_DIR"] = str(here / "logs")

    print(f"[MS25] system={system} radius={radius} max_neighbours={max_nbrs}")
    print(f"[MS25] config={args.finetuning_config}")
    print(f"[MS25] ensemble={args.pretrained_model_ensemble_path}")
    print(f"[MS25] modelname={modelname}  (should match dataset folder name)")

    run_finetune(dictionary_variables, args)