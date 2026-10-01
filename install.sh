#!/bin/bash
# Install hydragnn-gfm-finetuning and matbench (skipping scikit-learn)

pip install -e .

pip install matminer>=0.7.4 scipy>=1.9.0 monty>=2022.4.26
pip install matbench --no-deps
