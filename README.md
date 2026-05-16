# HydraGNN GFM Fine-Tuning for Materials

This repository provides tools and utilities for fine-tuning the HydraGNN Graph Foundation Model (GFM) ensemble on materials science datasets. The framework enables transfer learning from pre-trained graph neural network models to domain-specific tasks.

## Overview

The Graph Foundation Model (GFM) ensemble is a collection of pre-trained HydraGNN models that can be fine-tuned for various molecular and materials property prediction tasks. This repository includes:

- Utilities for fine-tuning model ensembles
- Example configurations for common datasets (QM9)
- Tools for model adaptation and head configuration
- Data preprocessing utilities

## Project Structure

```
├── README.md
├── HydraGNN/                         # HydraGNN install (gitignored, clone locally)
├── examples/
│   ├── abc3/
│   │   ├── abc3_getData_API.py       # ABC3 data download via API
│   │   ├── abc3_preonly.py           # ABC3 preprocessing script
│   │   ├── ensemble_fine_tune.py     # Fine-tuning script for ABC3
│   │   └── ensemble_fine_tune_sweep.py  # Hyperparameter sweep script
│   ├── matbench/
│   │   ├── matbench_preonly.py       # Matbench preprocessing script
│   │   ├── ensemble_fine_tune.py     # Fine-tuning script for Matbench
│   │   ├── finetuning_config.json    # Default config
│   │   └── finetuning_config_bce.json  # Binary cross-entropy config
│   ├── materials_project/
│   │   ├── materials_project_preonly.py  # Materials Project preprocessing
│   │   └── ensemble_fine_tune.py    # Fine-tuning script
│   ├── md17/
│   │   ├── md17_preonly.py           # MD17 preprocessing script
│   │   ├── md17_mlip_preonly.py      # MD17 MLIP preprocessing script
│   │   ├── ensemble_fine_tune.py     # Fine-tuning script for MD17
│   │   ├── run_benchmark.py          # Benchmark runner
│   │   └── param_count.py            # Parameter counting utility
│   ├── ms25/
│   │   ├── ms25_preonly.py           # MS25 preprocessing script
│   │   └── ensemble_fine_tune.py    # Fine-tuning script for MS25
│   ├── oqmd/
│   │   ├── oqmd_getData.py           # OQMD data download script
│   │   ├── oqmd_preonly.py           # OQMD preprocessing script
│   │   ├── ensemble_fine_tune.py     # Fine-tuning script for OQMD
│   │   └── ensemble_fine_tune_sweep.py  # Hyperparameter sweep script
│   ├── qm9/
│   │   ├── qm9_preonly.py            # QM9 preprocessing script
│   │   ├── qm9_energy_preonly.py     # QM9 energy-only preprocessing
│   │   ├── ensemble_fine_tune.py     # Fine-tuning script for QM9
│   │   └── run_benchmark.py          # Benchmark runner
│   └── wiggle150/
│       ├── wiggle150_preonly.py      # Wiggle150 preprocessing script
│       ├── ensemble_fine_tune.py     # Fine-tuning script for Wiggle150
│       ├── run_benchmark.py          # Benchmark runner
│       ├── benchmark_precision.py    # Precision benchmark utilities
│       └── evaluate_checkpoint.py   # Checkpoint evaluation script
└── utils/
    ├── __init__.py
    ├── ensemble_utils.py             # Core fine-tuning utilities
    ├── update_model.py               # Model architecture modification tools
    ├── evaluate_dataset.py           # Dataset evaluation utilities
    └── debug.py                      # Debugging utilities
```

## Installation

### Prerequisites

1. **HydraGNN v5.0**: Clone [HydraGNN v5.0](https://github.com/ORNL/HydraGNN/releases/tag/v5.0) directly into the project root (it is gitignored and kept local):
   ```bash
   cd /path/to/HydraGNN_GFM_FineTuning4Materials
   git clone --branch v5.0 https://github.com/ORNL/HydraGNN.git
   cd HydraGNN
   pip install -e .
   ```

2. **Python Dependencies**: All required dependencies are installed automatically with HydraGNN above.
   Follow any additional instructions in the [HydraGNN v5.0 release notes](https://github.com/ORNL/HydraGNN/releases/tag/v5.0) for your platform.

3. **Environment Setup**: Update your PYTHONPATH to include the project root:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials"
   ```

   Or add this to your `.bashrc` or `.zshrc`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials"
   ```

### Download Pre-trained Model Ensemble

Download the pre-trained GFM ensemble from [HuggingFace](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024):

```bash
# Download all model checkpoints and configuration files
# Each ensemble member will be fine-tuned independently
```

The model ensemble contains multiple pre-trained models with their respective configuration files organized in a structured directory format.

## Usage

### Environment Setup

**Important**: Before running any scripts, ensure your PYTHONPATH includes the project root:

```bash
# Option 1: Set temporarily for current session
export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials"

# Option 2: Add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials"' >> ~/.bashrc
source ~/.bashrc
```

### Quick Start with QM9 Example

1. **Navigate to the QM9 example directory**:
   ```bash
   cd examples/qm9/
   ```

2. **Prepare your dataset** (if not using QM9):
   - Prepare your data in the appropriate format
   - Update the feature schema in the fine-tuning script if needed

3. **Configure fine-tuning parameters**:
   - Modify `finetuning_config.json` to specify:
     - Output head architecture
     - Task weights
     - Layer dimensions
     - Number of tasks

4. **Run fine-tuning**:
   ```bash
   python ensemble_fine_tune.py
   ```

### Configuration

The fine-tuning process is controlled by JSON configuration files that specify:

- **Output Heads**: Define the architecture of task-specific prediction heads
- **Task Configuration**: Specify output dimensions, types, and weights
- **Training Parameters**: Learning rates, batch sizes, and optimization settings

Example configuration structure:
```json
{
    "NeuralNetwork": {
        "Architecture": {
            "output_heads": {
                "graph": [{
                    "type": "branch-0",
                    "architecture": {
                        "dim_pretrained": 50,
                        "num_sharedlayers": 2,
                        "dim_sharedlayers": 5,
                        "num_headlayers": 2,
                        "dim_headlayers": [50, 25]
                    }
                }]
            },
            "output_dim": [1],
            "output_type": ["graph"]
        }
    }
}
```

### Data Preparation

For custom datasets, ensure your data includes:

- **Graph Features**: Energy or other global molecular properties
- **Node Features**: Atomic numbers, coordinates, and other atomic properties
- **Proper Formatting**: The framework supports various data formats depending on your use case

The framework expects specific feature schemas that can be customized in the fine-tuning scripts. Data format requirements may vary based on your specific dataset and configuration.

## Key Components

### `utils/ensemble_utils.py`
Core utilities for ensemble fine-tuning including:
- Argument parsing for fine-tuning parameters
- Distributed training setup
- Model loading and configuration (supports both ensemble root dirs and single pretrained model dirs)
- Training loop management

### `utils/update_model.py`
Tools for modifying model architectures:
- Creating custom MLP heads for different tasks
- Adapting pre-trained models to new output dimensions
- Handling different prediction types (graph-level, node-level)

### `utils/evaluate_dataset.py`
Utilities for evaluating model performance on datasets.

### `utils/debug.py`
Debugging helpers for inspecting models and data during development.

### Example Scripts
Each dataset under `examples/` follows the same pattern:
- `*_preonly.py` — data download and preprocessing
- `ensemble_fine_tune.py` — main fine-tuning launcher
- `run_benchmark.py` — benchmark evaluation (where available)

| Dataset | Description |
|---|---|
| `abc3` | ABC3 perovskite-type compounds |
| `matbench` | Matbench materials benchmark suite |
| `materials_project` | Materials Project database |
| `md17` | MD17 molecular dynamics trajectories (also supports MLIP configs) |
| `ms25` | MS25 dataset |
| `oqmd` | Open Quantum Materials Database |
| `qm9` | QM9 molecular property prediction (also supports energy-only configs) |
| `wiggle150` | Wiggle150 benchmark dataset |

## Advanced Usage

### Custom Datasets

To use your own dataset:

1. Prepare data in the appropriate format for your use case
2. Define feature schema in your fine-tuning script
3. Create appropriate configuration JSON
4. Modify output heads to match your tasks

### Multi-Task Learning

The framework supports multi-task learning scenarios:
- Configure multiple output heads in the JSON configuration
- Specify task weights for balanced training
- Define different architectures for different task types

## Troubleshooting

### Common Issues

1. **Import Errors**: If you encounter `ModuleNotFoundError` for HydraGNN or project modules:
   - Verify your PYTHONPATH includes the project root:
     ```bash
     echo $PYTHONPATH
     ```
   - Check that the path is correct and the directory exists
   - For VS Code debugging, the PYTHONPATH is automatically configured in `.vscode/launch.json`

2. **Environment Variables**: Ensure you've sourced your shell profile after adding PYTHONPATH:
   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

3. **Virtual Environment**: If using a virtual environment, activate it before setting PYTHONPATH:
   ```bash
   source .venv/bin/activate
   export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials"
   ```

## Contributing

This project is part of the ORNL HydraGNN ecosystem. Contributions should follow the established patterns and maintain compatibility with the broader HydraGNN framework.

## License

This project follows the same license as HydraGNN. Please refer to the main HydraGNN repository for licensing information.

## Citation

If you use this code in your research, please cite HydraGNN v5.0:

```
Lupo Pasini, Massimiliano, Choi, Jong Youl, Mehta, Kshitij, Zhang, Pei, Weaver, Rylie,
Messerly, Richard, Chowdhury, Arindam, Raman, Adithya, & Aji, Ashwin M. (2026).
HydraGNN v5.0. https://doi.org/10.11578/dc.20260512.1
```

Available at:
- Release: https://github.com/ORNL/HydraGNN/releases/tag/v5.0
- DOE Code: https://www.osti.gov/biblio/code-180990

