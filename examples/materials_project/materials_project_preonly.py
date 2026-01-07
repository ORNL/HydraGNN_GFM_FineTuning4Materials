from __future__ import annotations

import os
import sys
from pathlib import Path
from itertools import combinations, islice
from typing import List
import math
import random

import torch
from mp_api.client import MPRester
from pymatgen.core import Structure, Element
from torch_geometric.data import Data
from torch_geometric.transforms import Distance

# Ensure the HydraGNN package is importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
HYDRAGNN_ROOT = REPO_ROOT / "HydraGNN"
if str(HYDRAGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDRAGNN_ROOT))

from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.datasets.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.utils.distributed import setup_ddp

# Set a default API key here if you prefer not to rely on the environment. Leaving it
# blank will make the script fall back to the MP_API_KEY environment variable.
API_KEY = "1b0OiUQGv3oBrPIl5Vv8kxs9QCXHjv23"

ELEMENT_POOL = [
    # Periods 1-4, excluding He, Ne, Ar, Kr, Be, Sc, Br, As, Se
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
]

ELEMENT_COUNTS = [1, 2, 3]  # explore pure elements through ternaries
DOWNSAMPLE_FRACTION = 1.0  # set <1.0 to use only a fraction for fine-tuning after download (balanced by composition)
DOWNSAMPLE_SEED = 1337  # seed to make downsampling repeatable
CACHE_PATH = os.path.join("dataset", "materials_project_screen.pt")  # cache all fetched Data objects
USE_CACHE = True  # skip API calls if CACHE_PATH exists

# Query/result limiting to keep API usage and runtime bounded.
RESULTS_CHUNK_SIZE = 200  # number of docs requested per page from the API
RESULTS_PER_QUERY_CAP = 400  # hard cap on docs consumed per element combination
RESULTS_PER_QUERY_SAMPLE = None  # optional random sample taken from capped results (set None to disable)

# Graph construction hyperparameters.
NEIGHBOR_RADIUS = 5.0
MAX_NEIGHBORS = 32
TARGET_FIELDS = [
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "efermi",
]


def _fmt_float(val: float | None) -> str:
    return f"{val:.3f}" if val is not None else "n/a"


def _get_api_key() -> str:
    key = API_KEY or os.getenv("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "Set MP_API_KEY in your environment or edit API_KEY in this script before running."
        )
    return key


def _doc_targets_to_tensor(doc) -> torch.Tensor:
    values = []
    for field in TARGET_FIELDS:
        val = getattr(doc, field, None)
        values.append(float("nan") if val is None else float(val))
    return torch.tensor(values, dtype=torch.float32)


def structure_to_data(
    structure: Structure,
    doc,
    radius: float = NEIGHBOR_RADIUS,
    max_neighbors: int = MAX_NEIGHBORS,
) -> Data:
    """Convert a pymatgen Structure plus Materials Project doc to a PyG Data object."""

    pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
    atomic_numbers = torch.tensor(structure.atomic_numbers, dtype=torch.float32).view(
        -1, 1
    )
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    pbc = torch.tensor(structure.lattice.pbc, dtype=torch.bool)

    node_features = torch.cat((atomic_numbers, pos), dim=1)

    unique_elements = sorted({el.symbol for el in structure.composition.elements})
    num_elements = len(unique_elements)

    data = Data(
        pos=pos,
        x=node_features,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        material_id=doc.material_id,
        formula=str(structure.composition.reduced_formula),
        element_set=unique_elements,
        num_elements=num_elements,
        y=_doc_targets_to_tensor(doc),
    )

    if data.pbc.any():
        graph_builder = RadiusGraphPBC(
            r=radius, loop=False, max_num_neighbors=max_neighbors
        )
    else:
        graph_builder = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_neighbors)

    data = graph_builder(data)
    data = Distance(norm=False, cat=False)(data)

    if not hasattr(data, "edge_shifts"):
        data.edge_shifts = torch.zeros(
            (data.edge_index.size(1), 3), dtype=torch.float32
        )

    return data


def _summarize_data_object(data: Data) -> str:
    return (
        f"{data.material_id:>12} | {data.formula:<12} | "
        f"nodes={data.num_nodes:>3} edges={data.edge_index.size(1):>4} | "
        f"targets="
        + ", ".join(_fmt_float(v.item()) for v in data.y.squeeze(0))
    )


def screen_materials() -> List[Data]:
    api_key = _get_api_key()
    data_objects: List[Data] = []

    rng = random.Random(DOWNSAMPLE_SEED)

    with MPRester(api_key) as mpr:
        for nelems in ELEMENT_COUNTS:
            combos_iter = combinations(ELEMENT_POOL, nelems)

            for combo in combos_iter:
                # Paginate and cap results to avoid unbounded sweeps; then optionally sample.
                results_iter = mpr.materials.summary.search(
                    elements=list(combo),
                    num_elements=nelems,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        *TARGET_FIELDS,
                    ],
                    chunk_size=RESULTS_CHUNK_SIZE,
                )

                capped_results = list(islice(results_iter, RESULTS_PER_QUERY_CAP))
                if RESULTS_PER_QUERY_SAMPLE is not None and len(capped_results) > RESULTS_PER_QUERY_SAMPLE:
                    capped_results = rng.sample(capped_results, k=RESULTS_PER_QUERY_SAMPLE)

                print(
                    f"\n=== {combo} (nelems={nelems}) | taking {len(capped_results)} docs (cap {RESULTS_PER_QUERY_CAP}, sample {RESULTS_PER_QUERY_SAMPLE}) ==="
                )

                for doc in capped_results:
                    structure = mpr.get_structure_by_material_id(doc.material_id)
                    data = structure_to_data(structure, doc)
                    data_objects.append(data)
                    print(_summarize_data_object(data))

    return data_objects


def _load_cached_data(cache_path: str) -> List[Data] | None:
    if USE_CACHE and os.path.isfile(cache_path):
        print(f"Loading cached materials from {cache_path}...")
        return torch.load(cache_path)
    return None


def _cache_data(data_list: List[Data], cache_path: str) -> None:
    if not USE_CACHE:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data_list, cache_path)
    print(f"Cached {len(data_list)} materials to {cache_path}.")


def _balance_key_chemistry(data: Data):
    """Return a class key based on chemical composition (element set)."""

    # Preferred: stored element_set
    element_set = getattr(data, "element_set", None)
    if element_set:
        return tuple(element_set)

    # Fallback: derive from atomic numbers if element_set is missing
    atomic_numbers = getattr(data, "atomic_numbers", None)
    if atomic_numbers is not None:
        try:
            zs = {int(z) for z in atomic_numbers.view(-1).tolist()}
            symbols = sorted(Element.from_Z(z).symbol for z in zs)
            return tuple(symbols)
        except Exception:
            pass

    return getattr(data, "num_elements", None)


def downsample_data_balanced(
    data_list: List[Data],
    fraction: float,
    seed: int | None = None,
    key_fn=_balance_key_chemistry,
) -> List[Data]:
    """Sample a fraction while keeping class balance based on chemical composition (default)."""

    if fraction >= 1.0 or len(data_list) <= 1:
        return data_list

    rng = random.Random(seed)
    buckets = {}
    for d in data_list:
        key = key_fn(d)
        buckets.setdefault(key, []).append(d)

    sampled: List[Data] = []
    for key, items in buckets.items():
        if len(items) == 1:
            sampled.extend(items)
            continue
        keep = max(1, math.ceil(len(items) * fraction))
        sampled.extend(rng.sample(items, k=keep))

    return sampled


def save_with_simple_pickle_writer(
    data_list: List[Data],
    basedir: str = os.path.join("dataset", "materials_project.pickle"),
    perc_train: float = 0.8,
) -> str:
    os.makedirs(os.path.dirname(basedir), exist_ok=True)

    trainset, valset, testset = split_dataset(
        dataset=data_list,
        perc_train=perc_train,
        stratify_splitting=False,
    )

    deg = gather_deg(trainset)

    attrs = {"pna_deg": deg}

    SimplePickleWriter(
        trainset,
        basedir,
        "trainset",
        use_subdir=True,
        attrs=attrs,
    )
    SimplePickleWriter(
        valset,
        basedir,
        "valset",
        use_subdir=True,
    )
    SimplePickleWriter(
        testset,
        basedir,
        "testset",
        use_subdir=True,
    )

    return basedir


if __name__ == "__main__":
    setup_ddp()
    data_list = _load_cached_data(CACHE_PATH)
    if data_list is None:
        data_list = screen_materials()
        _cache_data(data_list, CACHE_PATH)

    original_count = len(data_list)
    data_list = downsample_data_balanced(
        data_list, DOWNSAMPLE_FRACTION, DOWNSAMPLE_SEED
    )
    print(
        f"Using {len(data_list)} / {original_count} materials after downsampling (fraction={DOWNSAMPLE_FRACTION})."
    )
    print(f"\nConstructed {len(data_list)} PyG Data objects.")
    saved_dir = save_with_simple_pickle_writer(data_list)
    print(
        "Saved SimplePickle dataset. Load with SimplePickleDataset(basedir=\"%s\", label=\"trainset|valset|testset\")"
        % saved_dir
    )