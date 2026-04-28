#!/usr/bin/env python3
from pathlib import Path
import os
import json
from utils.ensemble_utils import build_arg_parser, run_finetune

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

VASP_SYSTEMS = ["MgO-2x2", "MgO-4x4", "HEA", "Reaction", "Zr-O"]
PICKLE_TAG = "mlip_peratom"


def run_system(system, here, repo, scratch=False, freeze=False):
    radius, max_nbrs = MS25_CUTOFFS[system]

    parser = build_arg_parser()
    args = parser.parse_args([])
    args.format = "pickle"
    args.ddstore = False
    args.ddstore_width = None
    args.shmem = False
    args.log = None
    args.batch_size = None
    args.train_from_scratch = scratch
    args.pretrained_model_ensemble_path = str((repo / "pretrained_model").resolve())
    args.datasetname = f"{system}_{PICKLE_TAG}"

    if scratch:
        suffix = "scratch"
    elif freeze:
        suffix = "frozen"
    else:
        suffix = "mlip"

    args.modelname = f"{system}_{suffix}_seed0"

    base_cfg_path = (here / "finetuning_config.json").resolve()
    with open(base_cfg_path, "r") as f:
        cfg = json.load(f)

    cfg["NeuralNetwork"]["Architecture"]["radius"] = float(radius)
    cfg["NeuralNetwork"]["Architecture"]["max_neighbours"] = int(max_nbrs)
    cfg["NeuralNetwork"]["Architecture"]["periodic_boundary_conditions"] = True
    cfg["NeuralNetwork"]["Training"]["num_epoch"] = 10
    cfg["NeuralNetwork"]["Training"]["train_from_scratch"] = scratch
    # freeze_mode is read by apply_freeze_mode in update_model.py
    cfg["NeuralNetwork"]["Training"]["freeze_mode"] = "message passing" if freeze else "None"

    patched_cfg_path = here / f"finetuning_config_patched_{system}_{suffix}.json"
    with open(patched_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    args.finetuning_config = str(patched_cfg_path.resolve())

    log_dir = here / "logs" / args.modelname
    os.makedirs(log_dir, exist_ok=True)
    os.environ["FINETUNING_LOG_DIR"] = str(log_dir)

    dictionary_variables = {
        "graph_feature_names": ["energy"],
        "graph_feature_dims":  [1],
        "node_feature_names":  ["atomic_number", "cartesian_coordinates"],
        "node_feature_dims":   [1, 3],
    }

    print(f"\n{'='*60}")
    print(f"[MS25] system={system}  scratch={scratch}  freeze={freeze}")
    print(f"[MS25] dataset={args.datasetname}.pickle")
    print(f"[MS25] modelname={args.modelname}")
    print(f"[MS25] config={args.finetuning_config}")
    print(f"{'='*60}")

    run_finetune(dictionary_variables, args)


if __name__ == "__main__":
    parser = build_arg_parser()
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo = here.parent.parent

    systems = [args.system] if args.system else VASP_SYSTEMS

    for system in systems:
        if system not in MS25_CUTOFFS:
            print(f"[SKIP] Unknown system: {system}")
            continue
        run_system(system, here, repo, scratch=args.scratch, freeze=args.freeze)