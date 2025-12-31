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
from pymatgen.core import Structure
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
    "Li",
    "Na",
    "Mg",
    "Ca",
    "Zn",
    "Al",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Mo",
    "W",
    "Nb",
    "Ta",
    "Cu",
]

ELEMENT_COUNTS = [1, 2, 3, 4, 5]  # explore pure elements through quinaries
MAX_PER_COMBO = 5  # set to None to fetch all entries per element-set
MAX_COMBOS_PER_N = 5  # set to None to explore all combinations per cardinality
SAMPLE_FRACTION = 1.0  # set <1.0 to randomly sample a fraction of results

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

    data = Data(
        pos=pos,
        x=node_features,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        material_id=doc.material_id,
        formula=str(structure.composition.reduced_formula),
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

    with MPRester(api_key) as mpr:
        for nelems in ELEMENT_COUNTS:
            combos_iter = combinations(ELEMENT_POOL, nelems)
            if MAX_COMBOS_PER_N is not None:
                combos_iter = islice(combos_iter, MAX_COMBOS_PER_N)

            for combo in combos_iter:
                results = mpr.materials.summary.search(
                    elements=list(combo),
                    num_elements=nelems,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        *TARGET_FIELDS,
                    ],
                    chunk_size=MAX_PER_COMBO if MAX_PER_COMBO is not None else 1000,
                    num_chunks=1 if MAX_PER_COMBO is not None else None,
                )

                results = list(results)
                if SAMPLE_FRACTION < 1.0 and len(results) > 0:
                    keep = max(1, math.ceil(len(results) * SAMPLE_FRACTION))
                    results = random.sample(results, k=keep)

                print(f"\n=== {combo} (nelems={nelems}) ===")
                for doc in results:
                    structure = mpr.get_structure_by_material_id(doc.material_id)
                    data = structure_to_data(structure, doc)
                    data_objects.append(data)
                    print(_summarize_data_object(data))

    return data_objects


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
    data_list = screen_materials()
    print(f"\nConstructed {len(data_list)} PyG Data objects.")
    saved_dir = save_with_simple_pickle_writer(data_list)
    print(
        "Saved SimplePickle dataset. Load with SimplePickleDataset(basedir=\"%s\", label=\"trainset|valset|testset\")"
        % saved_dir
    )