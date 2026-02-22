from utils.ensemble_utils import build_arg_parser, load_finetuning_config, load_datasets
from typing import List
from torch.utils.data import DataLoader
import numpy as np

# Load dataloaders and find the histogram of the atomic numbers
def get_dataset_histogram_Z(datasets: List, dictionary_variables: dict, max_elements: int = 120) -> np.ndarray:
    
    # Find the index of "atomic_number" based on the schema
    z_index = 0
    feature_names = dictionary_variables['node_feature_names']
    feature_dims = dictionary_variables['node_feature_dims']
    for name, dim in zip(feature_names, feature_dims):
        if name == "atomic_number": 
            break
        z_index += dim 

    # Initialize the main histogram
    atomic_number_histogram = np.zeros(max_elements, dtype=np.int64)

    # Stream through the datasets
    for dataset in datasets:
        for item in dataset:
            
            # In PyTorch Geometric/HydraGNN, node features are stored in `item.x`
            x = item.x
            
            # Safely handle both Torch Tensors and Numpy arrays
            if hasattr(x, 'cpu'):
                x = x.cpu().numpy()
                
            # Extract Z column and cast to integer
            z_vals = x[:, z_index].astype(int)
            
            # Count atoms in THIS specific graph 
            graph_hist = np.bincount(z_vals)
            
            # Pad the main histogram if this graph contains an element > max_elements
            if len(graph_hist) > len(atomic_number_histogram):
                atomic_number_histogram = np.pad(
                    atomic_number_histogram, 
                    (0, len(graph_hist) - len(atomic_number_histogram))
                )
            
            # Accumulate directly into the main histogram
            atomic_number_histogram[:len(graph_hist)] += graph_hist

    return atomic_number_histogram

# Load a dataset and find the histogram of the atomic numbers
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # The paths below assume that you are running this script from the root directory.
    args.pretrained_model_ensemble_path = './pretrained_model_test' #_ensemble'
    args.finetuning_config = './examples/oqmd/finetuning_config.json'
    args.datasetname = 'oqmd'

    # Load fine-tuning config
    ft_config = load_finetuning_config(args)

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

    # Load the dataset
    trainset, valset, testset = load_datasets(args, ft_config, dictionary_variables)

    # Evaluate the histogram of atomic numbers in the whole data set
    atomic_number_histogram = get_dataset_histogram_Z([trainset, valset, testset],dictionary_variables)

    for i, count in enumerate(atomic_number_histogram):
        if count > 0:
            print(f"Atomic Number {i}: {count} atoms")
    # print(atomic_number_histogram)