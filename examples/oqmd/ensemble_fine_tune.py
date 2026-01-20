#!/usr/bin/env python3

""" Run HydraGNN's main train/test/validate loop on the given dataset / model combination,
    refactored so the main flow is a callable function that accepts an `args` object.
"""

from utils.ensemble_utils import build_arg_parser, run_finetune
import shutil
import os
from pathlib import Path

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # The paths below assume that you are running this script from the root directory.
    args.pretrained_model_ensemble_path = './pretrained_model_ensemble'
    args.finetuning_config = './examples/oqmd/finetuning_config.json'
    args.datasetname = 'oqmd'
    args.modelname = 'oqmd'

    # Place all logs/checkpoints under the example folder
    example_dir = Path(__file__).parent
    os.environ["FINETUNING_LOG_DIR"] = str(example_dir / "logs")

    # ---- feature schema (explicit override) ----
    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "pos"]
    node_feature_dims = [1, 3]
    dictionary_variables = {}
    dictionary_variables['graph_feature_names'] = graph_feature_names
    dictionary_variables['graph_feature_dims'] = graph_feature_dims
    dictionary_variables['node_feature_names'] = node_feature_names
    dictionary_variables['node_feature_dims'] = node_feature_dims

    run_finetune(dictionary_variables, args)