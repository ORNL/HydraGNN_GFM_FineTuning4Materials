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
├── examples/
│   └── qm9/
│       ├── ensemble_fine_tune.py     # Main fine-tuning script for QM9
│       ├── finetuning_config.json    # Configuration for fine-tuning heads
│       └── qm9_preonly.py           # QM9 preprocessing script
└── utils/
    ├── __init__.py
    ├── ensemble_utils.py             # Core fine-tuning utilities
    └── update_model.py              # Model architecture modification tools
```

## Installation

### Prerequisites

1. **HydraGNN**: Install the latest version from the main branch:
   ```bash
   git clone https://github.com/ORNL/HydraGNN.git
   cd HydraGNN
   pip install -e .
   ```

2. **Python Dependencies**: Install the dependencies required by HydraGNN:
   - Follow the installation instructions in the [HydraGNN repository](https://github.com/ORNL/HydraGNN)
   - All required dependencies will be installed automatically when you install HydraGNN with `pip install -e .`

3. **Environment Setup**: Update your PYTHONPATH to include both directories:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials:/path/to/HydraGNN"
   ```
   
   Or add these lines to your `.bashrc` or `.zshrc`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials:/path/to/HydraGNN"
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

**Important**: Before running any scripts, ensure your PYTHONPATH includes both the HydraGNN_GFM_FineTuning4Materials and HydraGNN directories:

```bash
# Option 1: Set temporarily for current session
export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials:/path/to/HydraGNN"

# Option 2: Add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials:/path/to/HydraGNN"' >> ~/.bashrc
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
- Model loading and configuration
- Training loop management

### `utils/update_model.py`
Tools for modifying model architectures:
- Creating custom MLP heads for different tasks
- Adapting pre-trained models to new output dimensions
- Handling different prediction types (graph-level, node-level)

### Example Scripts
- `examples/qm9/ensemble_fine_tune.py`: Complete example for QM9 molecular property prediction
- `examples/qm9/qm9_preonly.py`: Data preprocessing utilities for QM9

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
   - Verify your PYTHONPATH includes both directories:
     ```bash
     echo $PYTHONPATH
     ```
   - Check that the paths are correct and the directories exist
   - For VS Code debugging, the PYTHONPATH is automatically configured in `.vscode/launch.json`

2. **Environment Variables**: Ensure you've sourced your shell profile after adding PYTHONPATH:
   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

3. **Virtual Environment**: If using a virtual environment, activate it before setting PYTHONPATH:
   ```bash
   source .venv/bin/activate
   export PYTHONPATH="${PYTHONPATH}:/path/to/HydraGNN_GFM_FineTuning4Materials:/path/to/HydraGNN"
   ```

## Contributing

This project is part of the ORNL HydraGNN ecosystem. Contributions should follow the established patterns and maintain compatibility with the broader HydraGNN framework.

## License

This project follows the same license as HydraGNN. Please refer to the main HydraGNN repository for licensing information.

## Citation

If you use this code in your research, please cite the relevant HydraGNN and GFM papers.

