#!/usr/bin/env python3

"""Run the oqmd fine-tuning sweep one case at a time and save per-case logs."""

from utils.ensemble_sweep import run_example_sweep


if __name__ == "__main__":
    run_example_sweep(
        example_name="oqmd",
        graph_feature_names=["energy"],
        graph_feature_dims=[1],
        node_feature_names=["atomic_number", "pos"],
        node_feature_dims=[1, 3],
    )
