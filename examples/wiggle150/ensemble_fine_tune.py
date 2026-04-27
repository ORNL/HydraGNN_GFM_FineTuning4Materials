#!/usr/bin/env python3

""" Run HydraGNN's main train/test/validate loop on the given dataset / model combination,
    refactored so the main flow is a callable function that accepts an `args` object.
"""

import sys
from pathlib import Path
import os

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.ensemble_utils import build_arg_parser, run_finetune, run_scratch

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # Place all logs/checkpoints under the example folder
    example_dir = Path(__file__).parent
    repo_root = example_dir.parents[1]

    if not getattr(args, "pretrained_model_ensemble_path", None) or args.pretrained_model_ensemble_path == "pretrained_model_ensemble":
        args.pretrained_model_ensemble_path = str((repo_root / "pretrained_model_ensemble").resolve())
    else:
        args.pretrained_model_ensemble_path = str(Path(args.pretrained_model_ensemble_path).expanduser().resolve())

    if not getattr(args, "finetuning_config", None) or args.finetuning_config == "./finetuning_config.json":
        args.finetuning_config = str((example_dir / "finetuning_config.json").resolve())
    else:
        args.finetuning_config = str(Path(args.finetuning_config).expanduser().resolve())

    if not getattr(args, "datasetname", None):
        args.datasetname = 'wiggle150'
    if not getattr(args, "modelname", None):
        args.modelname = 'wiggle150_scratch' if getattr(args, "train_from_scratch", False) else 'wiggle150'

    os.environ["FINETUNING_LOG_DIR"] = str(example_dir / "logs")

    # ---- feature schema (explicit override) ----
    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates"]
    node_feature_dims = [1, 3]
    dictionary_variables = {}
    dictionary_variables['graph_feature_names'] = graph_feature_names
    dictionary_variables['graph_feature_dims'] = graph_feature_dims
    dictionary_variables['node_feature_names'] = node_feature_names
    dictionary_variables['node_feature_dims'] = node_feature_dims

    if getattr(args, "train_from_scratch", False):
        run_scratch(dictionary_variables, args)
    else:
        run_finetune(dictionary_variables, args)
