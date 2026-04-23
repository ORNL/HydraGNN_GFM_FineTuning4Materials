#!/usr/bin/env python3
"""Run HydraGNN fine-tuning on QM9 (full fine-tune, frozen, or scratch)."""
from utils.ensemble_utils import build_arg_parser, run_finetune
import os
import json
from pathlib import Path

if __name__ == "__main__":
    parser = build_arg_parser()
    parser.add_argument("--freeze", action="store_true",
                        help="Freeze backbone (message passing layers).")
    args = parser.parse_args()

    args.pretrained_model_ensemble_path = './pretrained_model'
    args.datasetname = 'qm9'

    if args.train_from_scratch:
        if args.modelname is None:
            args.modelname = 'qm9_scratch_seed0'
    elif args.freeze:
        if args.modelname is None:
            args.modelname = 'qm9_frozen_seed0'
    else:
        if args.modelname is None:
            args.modelname = 'qm9_finetune_seed0'

    example_dir = Path(__file__).parent
    log_dir = example_dir / "logs" / args.modelname
    os.makedirs(log_dir, exist_ok=True)
    os.environ["FINETUNING_LOG_DIR"] = str(log_dir)

    # Patch freeze_mode into config
    base_cfg_path = example_dir / "finetuning_config.json"
    with open(base_cfg_path, "r") as f:
        cfg = json.load(f)

    cfg["NeuralNetwork"]["Training"]["train_from_scratch"] = args.train_from_scratch
    cfg["NeuralNetwork"]["Training"]["freeze_mode"] = "message passing" if args.freeze else "None"

    suffix = "scratch" if args.train_from_scratch else ("frozen" if args.freeze else "finetune")
    patched_cfg_path = example_dir / f"finetuning_config_patched_{suffix}.json"
    with open(patched_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    args.finetuning_config = str(patched_cfg_path.resolve())

    dictionary_variables = {
        'graph_feature_names': ["energy"],
        'graph_feature_dims':  [1],
        'node_feature_names':  ["atomic_number", "cartesian_coordinates"],
        'node_feature_dims':   [1, 3],
    }

    print(f"[QM9] modelname={args.modelname}")
    print(f"[QM9] datasetname={args.datasetname}")
    print(f"[QM9] scratch={args.train_from_scratch}")
    print(f"[QM9] freeze={args.freeze}")
    print(f"[QM9] config={args.finetuning_config}")

    run_finetune(dictionary_variables, args)