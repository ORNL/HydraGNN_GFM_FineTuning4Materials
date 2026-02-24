import os
import json
import subprocess
from pathlib import Path
from typing import List

GREEN_CHECK = u'\u2705'  # ✅
RED_CROSS = u'\u274C'    # ❌
SKIP_SYMBOL = u'\u23ED'  # ⏭
RESTORE_SYMBOL = u'\u21BA' # ↺

def print_model_sanity_check(model):
    '''
    Prints a sanity check of the model's parameters, showing which layers are trainable and which are frozen.
    This is especially useful after applying fine-tuning modifications to ensure that the intended layers are frozen and the new head layers are trainable.
    '''
    print("\n" + "="*80)
    print(f"{'LAYER NAME':<50} | {'SHAPE':<15} | {'TRAINABLE'}")
    print("-"*80)
    
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        status = "✅ YES" if param.requires_grad else "❌ NO "
        print(f"{name:<50} | {str(list(param.shape)):<15} | {status}")
        
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            
    print("-"*80)
    print(f"Total Trainable Params: {trainable_params:,}")
    print(f"Total Frozen Params:    {frozen_params:,}")
    print("="*80 + "\n")

def modify_config_epochs(config_path, num_epochs=2):
    """Reads a JSON, modifies 'epochs' keys to num_epochs, and returns original data."""
    if not config_path.exists():
        return None
    
    # Open the json file and load its content
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Create a deep copy to restore later
    original_data = json.loads(json.dumps(data)) 

    # Direct path update 
    data["NeuralNetwork"]["Training"]["num_epoch"] = num_epochs  

    # Write new json back to the file    
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return original_data

def run_test_suite_on_examples(dataset_names:List, config_names:List, function_names:List[List], skip_if_data_exists:List[List], debug_epochs:int = 2):
    # Base directory containing all example folders
    example_dir = Path("./examples") 
    
    # Base directory containing all dataset folders
    dataset_dir = Path("./dataset")

    # Loop over examples
    numExamples = len(dataset_names)
    for i in range(numExamples):
        # Make newline
        print("\nRunning test suite for example:", dataset_names[i])
        # Grab examples
        dataset_name = dataset_names[i]
        config_name = config_names[i]
        functions = function_names[i]
        skip_flags = skip_if_data_exists[i]
        # Check if dataset exists
        dataset_exists = False
        for f in dataset_dir.iterdir():
            try:
                f_name = f.name.split(".")[0]
                f_ext = f.name.split(".")[1]
            except:
                continue
            isDir = f.is_dir()
            nameExists = (f_name == dataset_name)
            goodExt = (f_ext in ["pickle", "adios"])
            if isDir and nameExists and goodExt:
                dataset_exists = True
                break
        # Set the config to debug
        config_path = example_dir / dataset_name / config_name
        original_config = modify_config_epochs(config_path, num_epochs=debug_epochs)
        # Now loop over functions
        for j, func in enumerate(functions):
            # If this function is for getting data and the dataset already exists, skip it
            if skip_flags[j] and dataset_exists:
                print(f"\tSkipping {func} because dataset {dataset_name} already exists. {SKIP_SYMBOL}")
                continue
            # Otherwise, run the function
            print(f"\tRunning {func} for dataset {dataset_name}...")
            func_path = example_dir / dataset_name / f"{func}.py"
            if func_path.exists():
                result = subprocess.run(
                    ["python", func_path], 
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"\t\tError running {func}: {result.stderr} {RED_CROSS}")
                else:
                    print(f"\t\tSuccessfully ran {func}. {GREEN_CHECK}")
            else:
                print(f"\t\tFunction script {func_path} does not exist. Skipping. {RED_CROSS}")
        # Set the config back to original
        if original_config is not None:
            with open(config_path, 'w') as f:
                json.dump(original_config, f, indent=4)
            print(f"\tRestored original config for {dataset_name}. {RESTORE_SYMBOL}")

if __name__ == "__main__":

    # Make the list of examples, with function names and if they should always be run (ex: getting data)
    abc_dataset = "abc3"
    abc_config = "finetuning_config.json"
    abc_functions = ["abc3_getData_API", "abc3_preonly", "ensemble_fine_tune"]
    abc_skipIfDataExists = [True, True, False]

    # More examples
    oqmd_dataset = "oqmd"
    oqmd_config = "finetuning_config.json"
    oqmd_functions = ["oqmd_getData", "oqmd_preonly", "ensemble_fine_tune"]
    oqmd_skipIfDataExists = [True, True, False]

    # More examples
    qm9_dataset = "qm9"
    qm9_config = "finetuning_config.json"
    qm9_functions = ["qm9_preonly", "ensemble_fine_tune"]
    qm9_skipIfDataExists = [True, False]

    # Make into list of all functions
    all_datasets = [abc_dataset, oqmd_dataset, qm9_dataset]
    all_config = [abc_config, oqmd_config, qm9_config]
    all_functions = [abc_functions, oqmd_functions, qm9_functions]
    all_skip_flags = [abc_skipIfDataExists, oqmd_skipIfDataExists, qm9_skipIfDataExists]
    run_test_suite_on_examples(all_datasets, all_config, all_functions, all_skip_flags)