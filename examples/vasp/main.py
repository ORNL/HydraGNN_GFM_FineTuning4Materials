'''
Example script of finetuning HydraGNN
'''

import os
from pathlib import Path
from utils.single_model_finetune import build_arg_parser, run_finetune

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # Get current working directory
    cwd = Path(__file__).resolve().parents[0]

    # Set path to pretrained directory
    args.pretrained_model_path = str(cwd / "pretrained_model")

    # Set path to finetuning settings
    args.finetuning_config = str(cwd / "finetune_config.json")

    # Set path to data directory
    args.data_dir = str(cwd / "dataset")

    # Set file name of data file
    args.datasetname = "NaZrCl_5A_20N_Eform_400_fp64_size_100"

    # Set model name
    args.modelname = "finetune_foundation_model"

    # Set sample step for downsampling (optional)
    args.sample_interval = 1

    # Unfreeze both MPNN/Convolution layers
    args.unfreeze_conv_layers = [0,1]

    # Select branches to finetune (omit if all_branches set to True)
    args.selected_branches = [5, 7]
    args.all_branches = False

    os.environ["FINETUNING_LOG_DIR"] = str(cwd / "logs")

    dictionary_variables = {
        "graph_feature_names": ["energy"],
        "graph_feature_dims": [1],
        "node_feature_names": ["atomic_number", "cartesian_coordinates"],
        "node_feature_dims": [1, 3],
    }
    
    run_finetune(dictionary_variables, args)
