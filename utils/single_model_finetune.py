"""Fine-tune selected decoder branches of one pretrained HydraGNN model.

Unlike :mod:`utils.ensemble_utils`, this module does not construct an ensemble.
The pretrained architecture is kept intact and batches are routed through the
selected ``branch-N`` decoders.  Decoder branches outside the selection are
frozen.  By default the backbone remains trainable; use ``--freeze-backbone``
to train only the selected branches' ``graph_shared`` and ``heads_NN`` modules.

Branches can be supplied by the public functions, the command line, or
``NeuralNetwork.Training.selected_branches``.  The legacy ``selected_branch``
setting remains supported.  Decoder dimensions in the fine-tuning config are
intentionally ignored because this utility fine-tunes existing pretrained
heads instead of replacing them.
"""

from __future__ import annotations
#--------------------------------------
# General Libraries
#--------------------------------------
import sys, os, argparse, copy, glob, json, pickle, random, re

from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

#--------------------------------------
# Pytorch Libraries
#--------------------------------------

import torch
import torch.distributed as dist

#--------------------------------------
# HydraGNN Functions
#--------------------------------------
import hydragnn
from hydragnn.train.train_validate_test import (
    resolve_precision,
)
from hydragnn.utils.distributed import (
    get_device,
    setup_ddp,
)

from utils.debug import print_model_sanity_check
from utils.ensemble_utils import get_distributed_model_find_unused


def build_arg_parser() -> argparse.ArgumentParser:
    """Build arguments for fine-tuning task"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pretrained-model-path",
        dest="pretrained_model_path",
        default="./pretrained_model",
        help="Directory containing config.json and a .pk checkpoint(s)",
    )
    parser.add_argument(
        "--finetuning-config",
        dest="finetuning_config",
        default="finetuning_config_mlip.json",
        help="Path of configuration file for finetuning settings",
    )
    parser.add_argument("--data-dir",
                        default="./dataset",
                        help="Directory contating dataset files in .pkl format")
    parser.add_argument("--datasetname",
                        default=None,
                        help="Name of the .pkl file used for finetuning <datasetname>.pkl")
    
    parser.add_argument("--modelname",
                        default="finetuned_model",
                        help="Given name to finetuned model. Used as log directory name")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path; otherwise one is discovered recursively and the one with highest epoch num is used",
    )
    parser.add_argument(
        "--selected-branches",
        type=int,
        nargs="+",
        default=None,
        help="Pretrained branch indices to fine-tune (e.g. [1] or [2,4,8])",
    )
    parser.add_argument(
        "--all-branches",
        action="store_true",
        help="Fine-tune every output branch in the pretrained model",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freezes MPNN layers while finetuning branches only",
    )

    parser.add_argument(
        "--unfreeze-conv-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Zero-based graph_convs block indices to train while the rest of "
            "the backbone stays frozen (for example: 1, or 0 1)"
        ),
    )
    parser.add_argument(
        "--sample-interval",
        dest="sample_interval",
        type=int,
        default=1,
        help=(
            "Interval for sampling data (e.g. --sample-interval = 10 -> Data is sampled every 10 points)"
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    '''
    Retrieves the underlying model when it has been wrapped by DistributedDataParallel
    else it returns the model
    '''
    return model.module if hasattr(model, "module") else model

def _decoder_owner(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the model object that owns the output decoder modules.

    HydraGNN models can have multiple wrapper levels. For example, the model
    may be wrapped by DistributedDataParallel and may contain another model
    under its ``model`` attribute. This function finds the object containing
    ``graph_shared`` and ``heads_NN``.

    Parameters
    ----------
    model : torch.nn.Module
        The original or wrapped HydraGNN model.

    Returns
    -------
    torch.nn.Module
        The module that owns the graph-level output heads.
    """

    # Remove a DDP-style wrapper, if one is present.
    model = _unwrap(model)

    # Some HydraGNN wrappers store the underlying network in model.model.
    wrapped = getattr(model, "model", None)

    # Use the inner model when it owns the graph decoder.
    if wrapped is not None and hasattr(wrapped, "graph_shared"):
        return wrapped

    # Otherwise, the supplied model owns the decoder directly.
    return model

def _normalize_branches(branches: Iterable[Any]) -> List[int]:
    """Return unique, non-negative branch indices while preserving order."""
    normalized = []
    for value in branches:
        if isinstance(value, str):
            match = re.fullmatch(r"branch-(\d+)", value)
            value = match.group(1) if match else value
        try:
            branch = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid branch {value!r}; expected an integer or 'branch-N'"
            ) from exc
        if branch < 0:
            raise ValueError("Branch indices must be non-negative")
        if branch not in normalized:
            normalized.append(branch)
    if not normalized:
        raise ValueError("At least one branch must be selected")
    return normalized

def _selected_branches(
    ft_config: Dict[str, Any], cli_values: Optional[Iterable[int]] = None
) -> List[int]:
    """Resolve a branch list from CLI values, training config, or head specs."""

    # CLI Branch IDs
    if cli_values is not None:
        return _normalize_branches(cli_values)

    # FT configuration Branch IDs
    training_config = ft_config["NeuralNetwork"].get("Training", {})
    configured = training_config.get("selected_branches")
    if configured is not None:
        if isinstance(configured, (str, int)):
            configured = [configured]
        return _normalize_branches(configured)

    legacy = training_config.get("selected_branch")
    if legacy is not None:
        return _normalize_branches([legacy])

    # Infered Branch IDs from FT configuration of neural network
    specs = (
        ft_config["NeuralNetwork"]
        .get("Architecture", {})
        .get("output_heads", {})
        .get("graph", [])
    )
    inferred = [spec.get("type", "") for spec in specs]
    if not inferred:
        raise ValueError(
            "Set NeuralNetwork.Training.selected_branches, pass "
            "--selected-branches, or define graph output-head entries."
        )
    return _normalize_branches(inferred)


def _find_checkpoint(model_dir: Path, explicit: Optional[str], ft_config: Dict[str, Any]) -> Path:
    '''
    Finds checkpoint candidates in folder model_dir and returns the checkpoint of the latest epoch
    '''
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    startfrom = ( 
        ft_config.get("NeuralNetwork", {})
        .get("Training", {})
        .get("startfrom")
    )
    if startfrom and str(startfrom).endswith(".pk"):
        candidates.extend((Path(startfrom), model_dir / str(startfrom)))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checkpoints = [Path(p) for p in glob.glob(str(model_dir / "**" / "*.pk"), recursive=True)]
    
    if not checkpoints:
        raise FileNotFoundError(f"No .pk checkpoint found below {model_dir}")
    else:
        print("Checkpoint Candidates")
        for ckpt in checkpoints:
            print(ckpt)
        print("-----------------------")
        sys.stdout.flush()

    def sort_key(path: Path) -> Tuple[int, float]:
        match = re.search(r"_epoch_(\d+)\.pk$", path.name)
        return (int(match.group(1)) if match else -1, path.stat().st_mtime)

    return max(checkpoints, key = sort_key)


def _match_module_prefix(state_dict: Dict[str, torch.Tensor], 
                         model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    '''
    Matches the keys of the model state dictionary and the checkpoint state dictionary
    '''
    model_has_module = any(k.startswith("module.") for k in model.state_dict())
    checkpoint_has_module = any(k.startswith("module.") for k in state_dict)
    if model_has_module == checkpoint_has_module:
        return OrderedDict(state_dict)
    if checkpoint_has_module:
        return OrderedDict(
            (k.removeprefix("module."), value) for k, value in state_dict.items()
        )
    return OrderedDict((f"module.{k}", value) for k, value in state_dict.items())


def _materialize_lazy_conditioning(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> None:
    """
    Create conditioning layers that HydraGNN builds initialize lazily.
    """
    target = _decoder_owner(model)
    stripped = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    graph_key = next(
        (k for k in stripped if k.endswith("graph_conditioner.0.weight")), None
    )
    if graph_key and hasattr(target, "_ensure_graph_conditioner"):
        if getattr(target, "graph_conditioner", None) is None:
            target.use_graph_attr_conditioning = True
            target._ensure_graph_conditioner(
                stripped[graph_key].shape[1], get_device()
            )

    concat_key = next(
        (
            k
            for k in stripped
            if k.endswith("graph_concat_projector.weight")
            or k.endswith("graph_concat_projector.0.weight")
        ),
        None,
    )
    if concat_key and hasattr(target, "_ensure_graph_concat_projector"):
        if getattr(target, "graph_concat_projector", None) is None:
            in_features = stripped[concat_key].shape[1]
            channel_dim = getattr(target, "hidden_dim", in_features)
            target.use_graph_attr_conditioning = True
            target.graph_attr_conditioning_mode = "concat_node"
            target._ensure_graph_concat_projector(
                graph_attr_dim=max(in_features - channel_dim, 1),
                channel_dim=channel_dim,
                device=get_device(),
            )

    pool_key = next(
        (k for k in stripped if k.endswith("graph_pool_projector.0.weight")), None
    )
    if pool_key and hasattr(target, "_ensure_graph_pool_projector"):
        if getattr(target, "graph_pool_projector", None) is None:
            in_features = stripped[pool_key].shape[1]
            channel_dim = getattr(target, "hidden_dim", in_features)
            target.use_graph_attr_conditioning = True
            target.graph_attr_conditioning_mode = "fuse_pool"
            target._ensure_graph_pool_projector(
                graph_attr_dim=max(in_features - channel_dim, 1),
                channel_dim=channel_dim,
                device=get_device(),
            )

def _load_pretrained_model(model_dir: Path,
                           checkpoint: Path,
                           precision: str,) -> Tuple[torch.nn.Module, Dict[str, Any], torch.dtype]:

    device = get_device()

    # Pretrained model configuration
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing pretrained config: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as stream:
        pretrained_config = json.load(stream)

    # Resolve precision differences
    _, param_dtype, _ = resolve_precision(precision)

    # Initialize model
    model = hydragnn.models.create_model_config(
        config=pretrained_config["NeuralNetwork"],
        verbosity=pretrained_config.get("Verbosity", {}).get("level", 1),
    )

    model = model.to(device=device, dtype=param_dtype)

    # Load Checkpoint
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint}")
    
    # Preprocess model and checkpoint
    _materialize_lazy_conditioning(model, state_dict)
    state_dict = _match_module_prefix(state_dict, model)
    model.load_state_dict(state_dict, strict=True)

    return model, pretrained_config, param_dtype


def _available_branches(model: torch.nn.Module) -> List[int]:
    """Return the graph branch indices provided by the pretrained model."""
    target = _decoder_owner(model)
    graph_shared = getattr(target, "graph_shared", None)
    if graph_shared is None:
        raise AttributeError("The pretrained model has no graph_shared decoders")
    branches = []
    for name in graph_shared.keys():
        match = re.fullmatch(r"branch-(\d+)", str(name))
        if match:
            branches.append(int(match.group(1)))
    if not branches:
        raise ValueError("The pretrained model has no branch-N graph decoders")
    return sorted(branches)


def _normalize_conv_indices(indices: Iterable[Any]) -> List[int]:
    """Return unique, non-negative message passing indices."""
    normalized = []
    for value in indices:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid convolution index {value!r}; expected an integer"
            ) from exc
        if index < 0:
            raise ValueError("Convolution indices must be non-negative")
        if index not in normalized:
            normalized.append(index)
    if not normalized:
        raise ValueError("At least one convolution index must be selected")
    return normalized


def _graph_convolution_blocks(
    model: torch.nn.Module,
) -> List[Tuple[str, torch.nn.Module]]:
    """Return the model's graph_convs blocks in forward order."""
    target = _decoder_owner(model)
    graph_convs = getattr(target, "graph_convs", None)
    if graph_convs is None:
        raise AttributeError(
            "The pretrained model has no graph_convs container"
        )
    blocks = [
        (f"graph_convs.{index}", module)
        for index, module in enumerate(graph_convs)
    ]
    if not blocks:
        raise ValueError("The pretrained model's graph_convs container is empty")
    return blocks


def _unfreeze_convolution_layers(
    model: torch.nn.Module, indices: Iterable[Any]
) -> List[str]:
    """Unfreeze selected graph_convs blocks and return their module names."""
    selected_indices = _normalize_conv_indices(indices)
    blocks = _graph_convolution_blocks(model)
    invalid = [index for index in selected_indices if index >= len(blocks)]
    available = ", ".join(
        f"{index}:{name}" for index, (name, _) in enumerate(blocks)
    )
    if invalid:
        raise IndexError(
            f"Convolution indices {invalid} are out of range. "
            f"Available layers are [{available}]"
        )

    selected_names = []
    for index in selected_indices:
        name, module = blocks[index]
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        selected_names.append(name)

    print(f"Available convolution layers (zero-based): [{available}]")
    print(
        "Unfrozen convolution layers: "
        + ", ".join(f"{index}:{blocks[index][0]}" for index in selected_indices)
    )
    return selected_names

def _configure_branches(
    model: torch.nn.Module,
    branches: Iterable[int],
    freeze_conv_layers: bool,
    unfreeze_conv_layers: Optional[Iterable[int]] = None,
) -> Tuple[int, int]:
    """Configure trainable decoders and optional graph-convolution blocks."""
    selected = {f"branch-{branch}" for branch in _normalize_branches(branches)}
    target = _decoder_owner(model)
    graph_shared = getattr(target, "graph_shared", None)
    heads = getattr(target, "heads_NN", None)
    available = set(graph_shared.keys()) if graph_shared is not None else set()
    missing_shared = selected - available
    if missing_shared:
        raise KeyError(
            f"Branches {sorted(missing_shared)} are absent from pretrained "
            f"graph_shared; available={sorted(available)}"
        )
    if heads is None:
        raise AttributeError("The pretrained model has no heads_NN decoders")

    # Reset stale checkpoint/config freezes, then apply the requested policy.
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze_conv_layers)

    for name, module in graph_shared.items():
        trainable = name in selected
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    found_heads = set()
    for head_dict in heads:
        for name, module in head_dict.items():
            trainable = name in selected
            if trainable:
                found_heads.add(name)
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
    missing_heads = selected - found_heads
    if missing_heads:
        raise KeyError(
            f"Branches {sorted(missing_heads)} are absent from pretrained heads_NN"
        )

    if unfreeze_conv_layers is not None:
        if not freeze_conv_layers:
            raise ValueError(
                "Explicit convolution selection requires a frozen backbone"
            )
        _unfreeze_convolution_layers(model, unfreeze_conv_layers)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    if trainable_count == 0:
        raise RuntimeError("The branch policy left no trainable parameters")
    return trainable_count, total_count

def _make_loaders(
    args: Any,
    ft_config: Dict[str, Any],
    branches: Iterable[int],
    dictionary_variables: Dict[str, Any],
):
    """Load, split, and route the dataset through each selected branch."""
    del dictionary_variables  # The pickled PyG objects already contain their schema.
    data_dir = Path(getattr(args, "data_dir", "./dataset")).expanduser()
    dataset_name = getattr(args, "datasetname")
    if not dataset_name:
        raise ValueError("args.datasetname is required (for example, 'NaZrCl_sdata')")

    data_path = data_dir / f"{dataset_name}.pkl"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Pass the filename stem as "
            "args.datasetname, without the .pkl extension."
        )
    with data_path.open("rb") as stream:
        dataset = pickle.load(stream)
    if not hasattr(dataset, "__len__") or len(dataset) < 3:
        raise ValueError(
            f"{data_path} must contain at least three PyG graph samples"
        )

    perc_train = float(ft_config["NeuralNetwork"]["Training"]["perc_train"])
    if not 0.0 < perc_train < 1.0:
        raise ValueError("NeuralNetwork.Training.perc_train must be between 0 and 1")

    seed = int(getattr(args, "seed", 1337))
    torch.manual_seed(seed)

    print("Loaded Dataset size: ", len(dataset))

    interval = getattr(args, "sample_interval")
    dataset = dataset[::interval]

    print("Subsampled dataset size: ", len(dataset))

    sys.stdout.flush()

    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        dataset, perc_train, False
    )

    branch_list = _normalize_branches(branches)

    def route_samples(samples):
        routed = []
        for data in samples:
            for branch in branch_list:
                branch_data = copy.deepcopy(data)
                branch_data.dataset_name = torch.tensor(
                    [[branch]], dtype=torch.long
                )
                routed.append(branch_data)
        return routed

    trainset = route_samples(trainset)
    valset = route_samples(valset)
    testset = route_samples(testset)

    print(
        f"Loaded {len(dataset)} samples from {data_path}; "
        f"routed to branches {branch_list}; split train/val/test="
        f"{len(trainset)}/{len(valset)}/{len(testset)}"
    )

    batch_size = getattr(args, "batch_size") or ft_config["NeuralNetwork"][
        "Training"
    ]["batch_size"]
    return hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, batch_size
    )

def _finetune_branches(
    dictionary_variables: Dict[str, Any],
    args: Any,
    branches: Optional[Iterable[int]],
    all_branches: bool,
) -> None:
    """Shared implementation using HydraGNN's built-in training routine."""
    
    model_dir_value = getattr(args, "pretrained_model_path", None)

    # Backward-compatible with the name used by the user's ensemble-based main.py.
    old_model_dir_value = getattr(args, "pretrained_model_ensemble_path",None)

    if old_model_dir_value and model_dir_value in (None, "./pretrained_model"):
        model_dir_value = old_model_dir_value
    model_dir = Path(model_dir_value or "./pretrained_model").expanduser().resolve()


    ft_config_path = Path(getattr(args, "finetuning_config")).expanduser().resolve()
    with ft_config_path.open("r", encoding="utf-8") as stream:
        ft_config = json.load(stream)

    verbosity = ft_config.get("Verbosity", {}).get("level", 1)
    training_config = ft_config["NeuralNetwork"]["Training"]
    arch_config = ft_config["NeuralNetwork"]["Architecture"]
    num_epochs = getattr(args, "num_epochs", None)
    
    if num_epochs is not None:
        if num_epochs <= 0:
            raise ValueError("args.num_epochs must be positive")
        training_config["num_epoch"] = num_epochs

    requested_branches = None
    if not all_branches:
        requested_branches = _selected_branches(ft_config, branches)

    with (model_dir / "config.json").open("r", encoding="utf-8") as stream:
        source_config = json.load(stream)

    source_precision = (
        source_config.get("NeuralNetwork", {})
        .get("Training", {})
        .get("precision", "fp32")
    )

    precision, param_dtype, _ = resolve_precision(
        training_config.get("precision", source_precision)
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(param_dtype)

    # Initalize DDP
    world_size, world_rank = setup_ddp()

    try:
        # Load Checkpoint        
        checkpoint = _find_checkpoint(
            model_dir, getattr(args, "checkpoint", None), ft_config
        )
        print("Loaded Checkpoint: ", checkpoint)

        # Load Checkpoint to Model
        model, _, param_dtype = _load_pretrained_model(
            model_dir, checkpoint, precision
        )
        # Acquire selected branches IDs
        selected_branches = (
            _available_branches(model) if all_branches else requested_branches
        )

        # Check for MPNN layer freezing
        unfreeze_conv_layers = getattr(args, "unfreeze_conv_layers", None)
        if unfreeze_conv_layers is None:
            unfreeze_conv_layers = training_config.get(
                "unfreeze_conv_layers",
                arch_config.get("unfreeze_conv_layers"),
            )
        if isinstance(unfreeze_conv_layers, int):
            unfreeze_conv_layers = [unfreeze_conv_layers]

        freeze_conv_layers = bool(
            getattr(args, "freeze_backbone", False)
            or arch_config.get("freeze_conv_layers", False)
            or unfreeze_conv_layers is not None
        )

        # Adjust trainable parameters
        trainable, total = _configure_branches(
            model,
            selected_branches,
            freeze_conv_layers,
            unfreeze_conv_layers,
        )

        # Setup model log dir
        hydragnn.utils.print.print_utils.setup_log(args.modelname)

        # Get Dist model
        model = get_distributed_model_find_unused(model, verbosity=verbosity)

        # Create dataloaders
        train_loader, val_loader, test_loader = _make_loaders(
            args, ft_config, selected_branches, dictionary_variables
        )

        # Create Optimizer
        optimizer_config = training_config["Optimizer"]

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr = optimizer_config["learning_rate"],
            weight_decay = optimizer_config.get("weight_decay", 0.0),
        )
        scheduler_config = training_config.get("Scheduler", {})
        factor = scheduler_config.get("factor",0.9)
        patience = scheduler_config.get("patience",10)
        threshold = scheduler_config.get("threshold",1e-3)
        cooldown = scheduler_config.get("cooldown",5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode="min", 
                                                                factor=factor, 
                                                                patience=patience, 
                                                                threshold=threshold, 
                                                                min_lr=1e-08,
                                                                cooldown=cooldown,
                                                                threshold_mode="rel")
    
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print("World Rank: ",world_rank)
        if rank == 0:
            print("World Size: ",world_size)
            print(f"Loaded checkpoint: {checkpoint}")
            print(f"Selected branches: {selected_branches}")
            

            # Print GPU(s) infomration
            if torch.cuda.is_available():
                print(f"Number of GPUs detected: {torch.cuda.device_count()}")
                print("Architecures Available: ", torch.cuda.get_arch_list())
                for i in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(i)
                    print(f"--- GPU {i} Specifications ---")
                    print(f"Name:            {prop.name}")
                    print(f"Compute Cap:     {prop.major}.{prop.minor}")
                    print(f"Total Memory:    {prop.total_memory / 1e9:.2f} GB")
                    print(f"Multi-Processors:{prop.multi_processor_count}\n")
                    print("-------------------------------")

            print_model_sanity_check(model)
            sys.stdout.flush()

        # Begin Finetuning
        # Writes out summaries and events
        writer = hydragnn.utils.model.model.get_summary_writer(args.modelname)

        saved_config = copy.deepcopy(source_config)
        saved_config["FineTuning"] = copy.deepcopy(ft_config["NeuralNetwork"])
        saved_config["FineTuning"]["selected_branches"] = selected_branches
        saved_config["FineTuning"]["all_branches"] = all_branches
        saved_config["FineTuning"]["unfreeze_conv_layers"] = (
            list(unfreeze_conv_layers)
            if unfreeze_conv_layers is not None
            else None
        )

        # Saving Configuration file with finetuning
        hydragnn.utils.input_config_parsing.save_config(saved_config, args.modelname)   

        hydragnn.train.train_validate_test(
            model = model,
            optimizer = optimizer,
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            writer = writer,
            scheduler = scheduler,
            config = ft_config["NeuralNetwork"],
            model_with_config_name = args.modelname,
            verbosity = verbosity,
            create_plots=False,
            compute_grad_energy=True,
            precision=training_config["precision"]
        )

        hydragnn.utils.model.save_model(model, optimizer, args.modelname)
        hydragnn.utils.profiling_and_tracing.print_timers(verbosity)

        if writer is not None:
            writer.close()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        torch.set_default_dtype(previous_dtype)


def finetune_selected_branches(
    dictionary_variables: Dict[str, Any],
    args: Any,
    branches: Optional[Iterable[int]] = None,
) -> None:
    """Fine-tune a selected list of pretrained decoder branches.

    ``branches`` takes priority over configuration values.  When it is omitted,
    branches are resolved from ``args.selected_branches``, the legacy
    ``args.selected_branch``, or the fine-tuning configuration.
    """
    if branches is None:
        branches = getattr(args, "selected_branches", None)
    if branches is None:
        legacy_branch = getattr(args, "selected_branch", None)
        if legacy_branch is not None:
            branches = [legacy_branch]
    _finetune_branches(dictionary_variables, args, branches, all_branches=False)


def finetune_all_branches(
    dictionary_variables: Dict[str, Any], args: Any
) -> None:
    """Fine-tune every graph decoder branch found in the pretrained model."""
    _finetune_branches(dictionary_variables, args, None, all_branches=True)

def run_finetune(dictionary_variables: Dict[str, Any], args: Any) -> None:
    """CLI-compatible dispatcher for all-branch or selected-branch training."""
    if getattr(args, "all_branches", False):
        if (
            getattr(args, "selected_branches", None) is not None
            or getattr(args, "selected_branch", None) is not None
        ):
            raise ValueError(
                "--all-branches cannot be combined with a selected branch option"
            )
        finetune_all_branches(dictionary_variables, args)
    else:
        finetune_selected_branches(dictionary_variables, args)
    
