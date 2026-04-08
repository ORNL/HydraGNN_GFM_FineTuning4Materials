import torch
import torch.nn as nn


def _build_mlp_layers(input_dim, hidden_dims, output_dim, activation_function):
    """Build a simple MLP with the model's activation between linear layers."""
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_function)
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _resolve_decoder_target(model):
    """Return the module whose forward path owns graph_shared and heads_NN."""
    wrapped_model = getattr(model, "model", None)
    if wrapped_model is not None and hasattr(wrapped_model, "graph_shared"):
        return wrapped_model
    return model


def _drop_wrapper_decoder_modules(model, decoder_target):
    """Remove stale wrapper decoder modules so parameter listings stay accurate."""
    if decoder_target is model:
        return

    for module_name in ("graph_shared", "heads_NN"):
        if module_name in model._modules:
            del model._modules[module_name]


def _freeze_module(module):
    """Disable gradients for every parameter owned by a module."""
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = False


def _normalize_freeze_mode(raw_mode):
    """Map user-facing freeze labels to canonical internal names."""
    normalized = str(raw_mode).strip().lower()
    aliases = {
        "none": "none",
        "shared + head": "shared_head",
        "shared+head": "shared_head",
        "shared_head": "shared_head",
        "message passing": "message_passing",
        "message_passing": "message_passing",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported freeze_mode. Use one of: 'None', 'shared + head', 'message passing'."
        )
    return aliases[normalized]


def _keep_pretrained_branch_zero(decoder_target):
    """Keep the loaded decoder weights and prune to the active branch-0 path."""
    if "branch-0" not in decoder_target.graph_shared:
        raise ValueError("Expected pretrained graph_shared to contain branch-0")

    graph_shared = nn.ModuleDict({"branch-0": decoder_target.graph_shared["branch-0"]})
    heads_nn = nn.ModuleList()
    for head_module in decoder_target.heads_NN:
        if "branch-0" not in head_module:
            raise ValueError("Expected pretrained heads_NN to contain branch-0")
        heads_nn.append(nn.ModuleDict({"branch-0": head_module["branch-0"]}))

    decoder_target.graph_shared = graph_shared
    decoder_target.heads_NN = heads_nn
    decoder_target.num_branches = 1

    if isinstance(decoder_target.config_heads, dict) and "graph" in decoder_target.config_heads:
        decoder_target.config_heads["graph"] = [
            branch_spec
            for branch_spec in decoder_target.config_heads["graph"]
            if branch_spec.get("type") == "branch-0"
        ]


def apply_freeze_mode(model, ft_config):
    """Freeze selected model regions for fine-tuning experiments."""
    decoder_target = _resolve_decoder_target(model)
    freeze_mode = _normalize_freeze_mode(
        ft_config["NeuralNetwork"]["Training"].get("freeze_mode", "none")
    )

    if freeze_mode == "none":
        return model

    if freeze_mode == "shared_head":
        _freeze_module(decoder_target.graph_shared)
        _freeze_module(decoder_target.heads_NN)
        return model

    if freeze_mode == "message_passing":
        for module_name in (
            "graph_convs",
            "feature_layers",
            "graph_conditioner",
            "graph_concat_projector",
            "graph_pool_projector",
            "pos_emb",
            "node_emb",
            "node_lin",
            "rel_pos_emb",
            "edge_emb",
            "edge_lin",
        ):
            _freeze_module(getattr(decoder_target, module_name, None))
        return model

    return model


def create_graph_shared_layers(config, hidden_dim, activation_function):
    """
    Create the shared graph-level decoder layers used before branch-specific heads.

    Args:
        config: Fine-tuning architecture configuration.
        hidden_dim: Backbone graph embedding size.
        activation_function: Activation module used by the base model.

    Returns:
        nn.ModuleDict: Shared graph decoder branches.
    """
    graph_shared = nn.ModuleDict({})

    for branch_spec in config["output_heads"].get("graph", []):
        branch_name = branch_spec["type"]
        branch_arch = branch_spec["architecture"]
        num_sharedlayers = branch_arch["num_sharedlayers"]
        dim_sharedlayers = branch_arch["dim_sharedlayers"]

        if num_sharedlayers < 1:
            raise ValueError("num_sharedlayers must be at least 1 for graph heads")

        shared_hidden_dims = [dim_sharedlayers] * num_sharedlayers
        graph_shared[branch_name] = _build_mlp_layers(
            hidden_dim,
            shared_hidden_dims[:-1],
            shared_hidden_dims[-1],
            activation_function,
        )

    return graph_shared


def create_mlps(config, activation_function):
    """
    Create branch-specific graph heads that consume the shared graph embedding.

    Args:
        config: Fine-tuning architecture configuration.
        activation_function: Activation module used by the base model.

    Returns:
        nn.ModuleList: One ModuleDict per output head.
    """
    output_dims = config["output_dim"]
    head_mlps = []

    for head_index, output_dim in enumerate(output_dims):
        head_mlp = nn.ModuleDict({})
        output_head_type = config["output_type"][head_index]
        output_head_specs = config["output_heads"].get(output_head_type, [])

        if not output_head_specs:
            raise ValueError(f"Missing output head specs for '{output_head_type}'")

        for branch_spec in output_head_specs:
            branch_name = branch_spec["type"]
            branch_arch = branch_spec["architecture"]
            dim_sharedlayers = branch_arch["dim_sharedlayers"]
            dim_headlayers = branch_arch["dim_headlayers"]

            head_mlp[branch_name] = _build_mlp_layers(
                dim_sharedlayers,
                dim_headlayers,
                output_dim,
                activation_function,
            )

        head_mlps.append(head_mlp)

    return nn.ModuleList(head_mlps)

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
    tot_loss = None
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
        weighted_loss = loss_i * weights[ihead].to(device=loss_i.device, dtype=loss_i.dtype)
        tot_loss = weighted_loss if tot_loss is None else tot_loss + weighted_loss
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
    decoder_target = _resolve_decoder_target(model)
    architecture_config = ft_config["NeuralNetwork"]["Architecture"]
    keep_pretrained_decoder = ft_config["NeuralNetwork"]["Training"].get(
        "keep_pretrained_decoder", False
    )

    if keep_pretrained_decoder:
        _keep_pretrained_branch_zero(decoder_target)
        _drop_wrapper_decoder_modules(model, decoder_target)
        print(decoder_target.graph_shared)
        print(decoder_target.heads_NN)
        return model

    device = next(decoder_target.parameters()).device

    decoder_target.graph_shared = create_graph_shared_layers(
        architecture_config,
        decoder_target.hidden_dim,
        decoder_target.activation_function,
    ).to(device)
    decoder_target.heads_NN = create_mlps(
        architecture_config,
        decoder_target.activation_function,
    ).to(device)
    decoder_target.config_heads = architecture_config["output_heads"]
    decoder_target.head_type = architecture_config["output_type"]
    decoder_target.head_dims = architecture_config["output_dim"]
    decoder_target.num_heads = len(decoder_target.heads_NN)
    decoder_target.num_branches = len(architecture_config["output_heads"].get("graph", []))

    _drop_wrapper_decoder_modules(model, decoder_target)
    print(decoder_target.graph_shared)
    print(decoder_target.heads_NN)

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
    model = apply_freeze_mode(model, ft_config)
    model = update_loss(model, ft_config)
    return model
