import torch.nn as nn


def create_mlps(config):
    """
    Creates head MLPs that take input from a pretrained shared layer.

    Args:
        config (dict): A dictionary with the following keys:
            - dim_pretrained (int): Dimension of the pretrained shared layer output.
            - dim_headlayers (list of int): List containing the dimensions for layers within each head.
            - output_dim (list of int): List containing the output dimensions for each head.

    Returns:
        list of nn.Sequential: A list of head MLP modules.
    """
    dim_pretrained = config["output_heads"]["graph"]["dim_pretrained"]
    dim_headlayers = config["output_heads"]["graph"]["dim_headlayers"]
    output_dims = config["output_dim"]

    # Create head MLPs
    head_mlps = []
    for head_index in range(len(output_dims)):
        head_layers = []
        in_dim = dim_pretrained  # Use pretrained layer dimension as input

        # Create hidden layers based on dim_headlayers
        for head_dim in dim_headlayers:
            head_layers.append(nn.Linear(in_dim, head_dim))
            head_layers.append(nn.ReLU())
            in_dim = head_dim

        # Output layer for each head with specified output dimension
        head_layers.append(nn.Linear(in_dim, output_dims[head_index]))

        # Create the head MLP as a Sequential model
        head_mlp = nn.Sequential(*head_layers)
        head_mlps.append(head_mlp)

    return nn.ModuleList(head_mlps)

def update_ensemble(model, ft_config):
    """
    Updates the given model ensemble according to the specified fine-tuning configuration.
    only meant to work on non-ddp models.
    Args:
        model (nn.Module): The model ensemble to be updated for fine-tuning. Must have `model_ens` attribute
        ft_config (dict): A configuration dictionary containing fine-tuning parameters
                          and architectural modifications.

    Returns:
        nn.Module: The updated ensemble model configured for fine-tuning.
    """
    updated_ens = [update_model(model_k) for model_k in model.model_ens]
    model.model_ens = updated_ens 
    model.num_heads = self.model_ens[0].module.num_heads
    model.loss = self.model_ens[0].module.loss
    model.model_size = len(self.model_dir_list)
    return model

def update_model(model, ft_config):
    """
    Updates the given model according to the specified fine-tuning configuration.
    only meant to work on non-ddp models.
    Args:
        model (nn.Module): The model to be updated for fine-tuning.
        ft_config (dict): A configuration dictionary containing fine-tuning parameters
                          and architectural modifications.

    Returns:
        nn.Module: The updated model configured for fine-tuning.
    """
    device = next(model.parameters()).device
    head_mlps = create_mlps(ft_config["FTNeuralNetwork"]["Architecture"])
#    print(model.heads_NN)
    # del model.heads_NN
    # Load existing state_dict
    state_dict = model.state_dict()
    # Filter out keys associated with 'heads_NN'
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('heads_NN')}
    # Reload state_dict into the model
    model.load_state_dict(filtered_state_dict, strict=False)

    # for name, param in list(model.named_parameters()):
    #     if name.startswith('heads_NN'):
    #         print(f'deleting {model} {name}')
    #         delattr(model, name)
    # if 'heads_NN' in model._modules:
    #     model._modules.pop('heads_NN')
    # for name, module in model.named_modules():
    #     print(name)
    model.heads_NN = head_mlps.to(device)
#    print(model.heads_NN)
    model.head_type = ft_config["FTNeuralNetwork"]["Architecture"]["output_heads"][
        "graph"
    ]["num_headlayers"] * ["graph"]
    model.head_dims = ft_config["FTNeuralNetwork"]["Architecture"]["output_dim"]
    model.num_heads = len(head_mlps)

    return model
