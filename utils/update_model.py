import torch
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

    output_dims = config["output_dim"]

    # Create head MLPs
    head_mlps = []

    for output_head_type, output_head_specs in config["output_heads"].items():

        if output_head_type == 'node' and output_head_specs[0]['architecture']:
            raise ValueError("Invalid input: Fine tuning for node-level prediction heads with convolutional layers not supported yet")

        dim_pretrained = output_head_specs[0]["architecture"]["dim_pretrained"]
        dim_headlayers = output_head_specs[0]["architecture"]["dim_headlayers"]

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
            head_mlp = nn.ModuleDict({})
            head_mlp['branch-0'] = nn.Sequential(*head_layers)
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

def update_loss(model, ft_config):
    """
    Updates the given model losses according to the specified fine-tuning configuration.
    only meant to work on non-ddp models.
    Args:
        model (nn.Module): The model to be updated for fine-tuning.
        ft_config (dict): A configuration dictionary containing fine-tuning parameters
                          and architectural modifications.

    Returns:
        nn.Module: The updated model configured for fine-tuning.
    """
    losses, weights = get_losses_and_weights(ft_config)
    model.loss = lambda pred, value, head_index: generic_loss(pred, value, head_index, losses, weights)
    return model

def generic_loss(pred, value, head_index, losses, weights):
    tot_loss = 0
    tasks_loss = []
    for ihead,loss in enumerate(losses):
        head_pre = pred[ihead]
        pred_shape = head_pre.shape
        head_val = value[head_index[ihead]]
        value_shape = head_val.shape
        # a bit dubious?
        if pred_shape != value_shape:
            head_val = torch.reshape(head_val, pred_shape)
        loss_i = loss(head_pre, head_val)
        tot_loss += (loss_i * weights[ihead]).float() 
        tasks_loss.append(tot_loss)
    return tot_loss, tasks_loss
        
def get_losses_and_weights(ft_config):
    losses = []
    for loss_type in ft_config['NeuralNetwork']['Training']["loss_function_types"]:
        if loss_type == 'regression':
            losses.append(nn.MSELoss())
        elif loss_type == 'mae':
            losses.append(nn.L1Loss())
        elif loss_type == 'binary':
            losses.append(nn.BCEWithLogitsLoss())
        elif loss_type == 'categorical':
            losses.append(nn.CrossEntropyLoss())
    weights = torch.FloatTensor(ft_config['NeuralNetwork']["Architecture"]["task_weights"] )
    return losses, weights

def update_architecture(model, ft_config):
    """
    Updates the given model architecture according to the specified fine-tuning configuration.
    only meant to work on non-ddp models.
    Args:
        model (nn.Module): The model to be updated for fine-tuning.
        ft_config (dict): A configuration dictionary containing fine-tuning parameters
                          and architectural modifications.

    Returns:
        nn.Module: The updated model configured for fine-tuning.
    """
    device = next(model.parameters()).device
    head_mlps = create_mlps(ft_config["NeuralNetwork"]["Architecture"])
    state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('heads_NN')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.heads_NN = head_mlps.to(device)

    model.head_type = []
    for output_head_type, output_head_specs in ft_config["NeuralNetwork"]["Architecture"]["output_heads"].items():
        model.head_type.append(output_head_type)

    model.head_dims = ft_config["NeuralNetwork"]["Architecture"]["output_dim"]
    model.num_heads = len(head_mlps)
    print(model.heads_NN)

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
    model = update_architecture(model, ft_config)        
    model = update_loss(model, ft_config)
    return model
