from utils.ensemble_utils import build_arg_parser, get_ensemble, load_finetuning_config, load_datasets, make_dataloaders, setup_distributed_finetuning, test_ensemble, test_ensemble_jamie
from typing import List
from torch.utils.data import DataLoader
import numpy as np

# Load datasets and find the histogram of the atomic numbers
def get_dataset_histogram_Z(trainset, valset, testset, dictionary_variables: dict, max_elements: int = 120) -> np.ndarray:
    
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
    for dataset in [trainset, valset, testset]:
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

# Load datasets, make dataloaders
def get_mean_deviation(trainset, valset, testset, dictionary_variables: dict, ft_config:dict, batch_size:int=16, ddstore:bool=False) -> np.ndarray:
    
    # Setup distributed finetuning environment
    comm_size, rank, comm = setup_distributed_finetuning()

    # Make dataloader for dataset
    train_loader, val_loader, test_loader = make_dataloaders(trainset, valset, testset, batch_size=batch_size,ddstore=ddstore)
    
    # Get the ensemble of models
    model = get_ensemble(args, ft_config, train_loader, val_loader, test_loader, freeze_conv=True)

    # Get results for each loader
    tr_y, tr_mu, tr_std = test_ensemble_jamie(model, train_loader, "train",  verbosity=2, save_results=False)
    va_y, va_mu, va_std = test_ensemble_jamie(model, val_loader, "val",  verbosity=2, save_results=False)
    te_y, te_mu, te_std = test_ensemble_jamie(model, test_loader, "test",  verbosity=2, save_results=False)

    # Find number of heads and concatenate the stds across datasets for each head
    num_heads = len(tr_std)
    all_heads_combined_std = []

    # Loop through each head and concatenate the datasets
    for i in range(num_heads):
        # Join train, val, and test for this specific head
        combined_for_this_head = np.concatenate([tr_std[i], va_std[i], te_std[i]])
        all_heads_combined_std.append(combined_for_this_head)
    
    # Or if you want the mean deviation per head:
    mean_std_per_head = [np.mean(h) for h in all_heads_combined_std]
    
    return mean_std_per_head
    
# Load a dataset and find the histogram of the atomic numbers
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # The paths below assume that you are running this script from the root directory.
    args.pretrained_model_ensemble_path = './pretrained_model_ensemble'
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

    # Load the datasets
    trainset, valset, testset = load_datasets(args, ft_config, dictionary_variables)

    # Evaluate the histogram of atomic numbers in the whole data set
    atomic_number_histogram = get_dataset_histogram_Z(trainset, valset, testset,dictionary_variables)

    # Evaluate the models on the dataset to fin 
    for i, count in enumerate(atomic_number_histogram):
        if count > 0:
            print(f"Atomic Number {i}: {count} atoms")

    # Evaluate the mean deviation of the ensemble's predictions across the datasets
    mean_std_per_head = get_mean_deviation(trainset, valset, testset, dictionary_variables, ft_config, batch_size = ft_config["NeuralNetwork"]["Training"]["batch_size"])
    print(f"Mean uncertainty per head: {mean_std_per_head}")