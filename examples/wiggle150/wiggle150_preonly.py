import os
import pdb
import json
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance
import argparse
import urllib.request
import shutil
import re
from pathlib import Path

import logging
import sys
from mpi4py import MPI

info = logging.info

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
import hydragnn.utils.profiling_and_tracing.tracer as tr

from hydragnn.utils.datasets.pickledataset import SimplePickleWriter
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.adiosdataset import AdiosWriter
except ImportError:
    AdiosWriter = None

WIGGLE150_URL = "https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.5c00015/suppl_file/ct5c00015_si_003.xyz"
# Default to ../datasets/wiggle150/raw at repo root
RAW_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "wiggle150" / "raw"
RAW_XYZ_PATH = RAW_DATASET_DIR / "wiggle150.xyz"
RADIUS_CUTOFF = 6.0
MAX_NUM_NEIGHBORS = 128

transform_coordinates = Distance(norm=False, cat=False)

try:
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
except ImportError:
    ASE_ATOMIC_NUMBERS = None


def element_symbol_to_atomic_number(symbol):
    """Map an element symbol to its atomic number.

    Tries ASE first; falls back to a minimal built-in dictionary for common elements.
    """
    s = symbol.strip().capitalize()
    if ASE_ATOMIC_NUMBERS is not None and s in ASE_ATOMIC_NUMBERS:
        return ASE_ATOMIC_NUMBERS[s]

    fallback = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
    }
    if s in fallback:
        return fallback[s]
    raise ValueError(f"Unknown element symbol: {symbol}")


def parse_energy_from_comment(comment):
    """Extract a single float energy from the XYZ comment line if present."""
    matches = re.findall(r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)", comment)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def download_wiggle150_xyz(url=WIGGLE150_URL, out_path=RAW_XYZ_PATH):
    """Download the wiggle150 XYZ file if missing."""
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    existing_xyz = list(RAW_DATASET_DIR.glob("*.xyz"))
    if existing_xyz:
        return existing_xyz[0]

    # Fallback: repository-level dataset directory (singular) that may already contain the file
    alt_dir = Path(__file__).resolve().parents[2] / "dataset" / "wiggle150" / "raw"
    alt_xyz = list(alt_dir.glob("*.xyz"))
    if alt_xyz:
        info(f"Found local wiggle150 data at {alt_xyz[0]}")
        return alt_xyz[0]

    # Allow overriding URL via env for mirrors
    final_url = os.environ.get("WIGGLE150_URL", url)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/plain,application/octet-stream,application/x-xyz,*/*;q=0.8",
        "Referer": "https://pubs.acs.org/",
    }
    req = urllib.request.Request(final_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download wiggle150 from {final_url}: {exc}. "
            "If access is blocked, download manually and place it at "
            f"{out_path}"
        ) from exc

    with open(out_path, "rb") as f:
        head = f.read(512).lower()
        if b"html" in head:
            raise RuntimeError(
                f"Downloaded content from {final_url} looks like HTML (likely access blocked). "
                f"Please download manually and place it at {out_path}"
            )
    return out_path


def load_wiggle150_xyz(xyz_path):
    """Parse multi-frame XYZ into torch_geometric Data objects."""
    dataset = []
    with open(xyz_path, "r") as f:
        while True:
            first = f.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            try:
                natoms = int(first)
            except ValueError:
                raise RuntimeError(f"Expected atom count, got: {first}")

            comment = f.readline().strip()
            z_list = []
            pos_list = []
            for _ in range(natoms):
                line = f.readline()
                if not line:
                    raise RuntimeError("Unexpected EOF while reading atoms")
                parts = line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"Malformed atom line: {line}")
                z_list.append(element_symbol_to_atomic_number(parts[0]))
                pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

            energy = parse_energy_from_comment(comment)
            y = torch.tensor([energy if energy is not None else 0.0], dtype=torch.float)

            # Concatenate atomic numbers and positions into x
            atomic_nums = torch.tensor(z_list, dtype=torch.float).view(-1, 1)
            positions = torch.tensor(pos_list, dtype=torch.float)
            x = torch.cat((atomic_nums, positions), dim=1)

            data = Data(
                x=x,
                pos=positions,
                y=y,
            )
            dataset.append(data)
    return dataset


def apply_graph_transforms(dataset, radius=RADIUS_CUTOFF, max_num_neighbors=MAX_NUM_NEIGHBORS):
    """Build radius graph and edge distances for each Data sample."""
    radius_graph = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_num_neighbors)
    processed = []
    for data in dataset:
        data = radius_graph(data)
        data = transform_coordinates(data)
        processed.append(data)
    return processed


def preview_wiggle150_xyz(url=WIGGLE150_URL, lines=40):
    """Stream and print the first `lines` lines from the wiggle150 XYZ source."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            for _ in range(lines):
                line = resp.readline()
                if not line:
                    break
                print(line.decode("utf-8", errors="ignore").rstrip("\n"))
    except Exception as exc:
        print(f"Failed to preview dataset from {url}: {exc}")


def main():
    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument(
        "--pretrained_model_ensemble_path", help="directory for ensemble of models", type=str, default="pretrained_model_ensemble"
    )
    parser.add_argument(
        "--dataset_url",
        help="Override wiggle150 XYZ download URL (else uses default or WIGGLE150_URL env)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--xyz_path",
        help="Path to a local wiggle150 XYZ file. If provided and exists, download is skipped.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--force_rebuild",
        help="Force rebuilding pickles even if they already exist.",
        action="store_true",
    )

    parser.add_argument(
        "--finetuning_config", help="path to JSON file with configuration for fine-tunable architecture", type=str,
        default="./finetuning_config.json"
    )
    parser.add_argument("--log", help="log name")
    parser.add_argument("--modelname", help="model name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    parser.add_argument(
        "--preview_dataset",
        help="Print first 40 lines of the remote wiggle150 XYZ (curl -L ... | head -n 40)",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preview_dataset:
        preview_wiggle150_xyz()
        sys.exit(0)

    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    modelname = "wiggle150" if args.modelname is None else args.modelname

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning_config.json")
    with open(filename, "r") as f:
        ft_config = json.load(f)

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates"]
    node_feature_dims = [1, 3]

    verbosity = ft_config["Verbosity"]["level"]
    var_config = ft_config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    log_name = "wiggle150_finetuning"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # If pickled dataset already exists and rebuild not forced, skip processing
    pickle_base = os.path.join(
        os.path.dirname(__file__), "../../dataset", f"{modelname}.pickle"
    )
    if (not args.force_rebuild) and os.path.isdir(pickle_base):
        has_train = os.path.isdir(os.path.join(pickle_base, "trainset"))
        has_val = os.path.isdir(os.path.join(pickle_base, "valset"))
        has_test = os.path.isdir(os.path.join(pickle_base, "testset"))
        if has_train and has_val and has_test:
            info(f"Found existing pickled dataset at {pickle_base}; skipping download and rebuild.")
            sys.exit(0)

    # Resolve raw XYZ path: prefer explicit local path, else download (with optional URL override)
    if args.xyz_path and os.path.isfile(args.xyz_path):
        raw_xyz = Path(args.xyz_path)
    else:
        download_url = args.dataset_url if args.dataset_url else WIGGLE150_URL
        raw_xyz = download_wiggle150_xyz(url=download_url)
    dataset = load_wiggle150_xyz(raw_xyz)
    dataset = apply_graph_transforms(dataset)

    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        dataset, ft_config["NeuralNetwork"]["Training"]["perc_train"], False
    )

    print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

    print("Before COMM.Barrier()", flush=True)
    comm.Barrier()
    print("After COMM.Barrier()", flush=True)

    deg = gather_deg(trainset)
    ft_config["pna_deg"] = deg

    ## adios
    if args.format == "adios":
        if AdiosWriter is None:
            raise ImportError("AdiosWriter not available; install ADIOS support or use --pickle")
        fname = os.path.join(
            os.path.dirname(__file__), "./dataset/%s.bp" % modelname
        )
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

    ## pickle
    elif args.format == "pickle":
        basedir = os.path.join(
            os.path.dirname(__file__), "../../dataset", "%s.pickle" % modelname
        )
        attrs = dict()
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
            attrs=attrs,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
    sys.exit(0)


if __name__ == "__main__":

    main()