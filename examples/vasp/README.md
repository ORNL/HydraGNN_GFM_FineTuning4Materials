# Finetuning a HydraGNN model on the VASP example

This example finetunes one or more existing branches from a pretrained
HydraGNN model on a pickled collection of PyTorch Geometric graphs. Finetuning
is launched by [`main.py`](main.py), while [`finetune_config.json`](finetune_config.json)
controls the optimizer, scheduler, loss, data split, and other training settings.

## 1. Prepare the environment

Install HydraGNN and its dependencies as described in the repository-level
README. From the HydraGNN repository root, make both HydraGNN and the GFM
utilities importable:

```bash
export PYTHONPATH="$PWD/GFM:$PWD:${PYTHONPATH}"
cd GFM/examples/vasp
```

The commands below assume they are run from `GFM/examples/vasp`. Running from
this directory also keeps the generated `logs/` directory beside this README.

## 2. Arrange the input files

The example expects the following layout:

```text
vasp/
├── main.py
├── finetune_config.json
├── dataset/
│   └── NaZrCl_5A_20N_Eform_400_fp64_size_100.pkl
└── pretrained_model/
    ├── config.json
    └── <checkpoint>_epoch_<N>.pk
```

The pretrained model directory must contain its original `config.json` and at
least one `.pk` checkpoint. Unless an explicit checkpoint is supplied, the
finetuning utility searches this directory recursively and selects the
checkpoint with the largest `_epoch_N` suffix.

The dataset must be a pickle file containing at least three PyTorch Geometric
graph samples. `datasetname` is always the filename **without** `.pkl`. For the
included dataset, use:

```python
args.datasetname = "NaZrCl_5A_20N_Eform_400_fp64_size_100"
```

## 3. Quick start: edit `main.py`

The simplest workflow is to edit the argument assignments in `main.py` and run
the script:

```python
args.pretrained_model_path = str(cwd / "pretrained_model")
args.finetuning_config = str(cwd / "finetune_config.json")
args.data_dir = str(cwd / "dataset")
args.datasetname = "NaZrCl_5A_20N_Eform_400_fp64_size_100"
args.modelname = "finetune_foundation_model"

# Use every sample. A value of 10 would retain every tenth sample.
args.sample_interval = 1

# Train decoder branches 5 and 7. Branch indices are zero-based.
args.selected_branches = [5, 7]
args.all_branches = False

# Unfreeze convolution blocks 0 and 1, if only branch finetune is needed omit the line below.
args.unfreeze_conv_layers = [0, 1]
```

Then run:

```bash
python main.py
```

To finetune every pretrained decoder branch instead, use:

```python
args.selected_branches = None
args.all_branches = True
```

Do not select explicit branches and enable `all_branches` at the same time.

### Feature names

The `dictionary_variables` block in `main.py` describes the features stored in
the graph objects:

```python
dictionary_variables = {
    "graph_feature_names": ["energy"],
    "graph_feature_dims": [1],
    "node_feature_names": ["atomic_number", "cartesian_coordinates"],
    "node_feature_dims": [1, 3],
}
```

Adjust these names and dimensions when adapting the launcher to a differently
prepared dataset. The finetuning loader uses the schema already stored in the
pickled graph objects, so the data and the output definitions in the JSON must
still be mutually consistent.

## 4. Configure training with `finetune_config.json`

The most commonly changed settings are under `NeuralNetwork.Training`:

```json
{
  "NeuralNetwork": {
    "Training": {
      "num_epoch": 100,
      "perc_train": 0.8,
      "batch_size": 25,
      "loss_function_type": "mae",
      "loss_function_types": ["mae"],
      "precision": "fp64",
      "selected_branches": [5, 7],
      "unfreeze_conv_layers": [0, 1],
      "Optimizer": {
        "type": "AdamW",
        "learning_rate": 0.00063368317
      },
      "Scheduler": {
        "type": "plateau",
        "factor": 0.5,
        "patience": 5,
        "threshold": 0.0001,
        "cooldown": 0
      }
    }
  }
}
```

Important configuration notes:

- Put `selected_branches` under `NeuralNetwork.Training`. The utility does not
  read it from the top level of `NeuralNetwork`.
- To let JSON choose the branches or convolution blocks, remove the matching
  `args.selected_branches` and `args.unfreeze_conv_layers` assignments from
  `main.py`, or set them to `None`.
- `perc_train` must be between 0 and 1. HydraGNN uses it when constructing the
  train, validation, and test splits.
- `Variables_of_interest` identifies the prediction targets and their locations
  in each graph. Update `output_names`, `output_index`, `output_dim`, and `type`
  for a different task.
- This utility preserves the pretrained decoders. Decoder dimensions written in
  `Architecture.output_heads` do not replace the existing pretrained heads.
- `Architecture.freeze_conv_layers: true` freezes the backbone. Supplying
  `unfreeze_conv_layers` also freezes the backbone first, then re-enables only
  the listed zero-based `graph_convs` blocks.
- `all_branches` is a launcher/CLI option, not a JSON setting.

Each graph is copied and routed through every selected branch. Selecting two
branches therefore produces twice as many routed train/validation/test samples
as selecting one branch.

## 5. Use command-line arguments

View all available options with:

```bash
python main.py --help
```

For example:

```bash
python main.py \
  --pretrained-model-dir pretrained_model \
  --num-epochs 20 \
  --batch-size 16 \
  --checkpoint pretrained_model/my_model_epoch_97.pk \
  --seed 42
```

`--num-epochs` and `--batch-size` override their JSON values. An explicit
`--checkpoint` takes priority over automatic checkpoint discovery.

### Allow every launcher setting to be overridden from the CLI

The supplied `main.py` parses the command line and then assigns example values.
Those later assignments win over CLI flags with the same destination. For a
CLI-first workflow, replace the parser and assignment section with the pattern
below:

```python
cwd = Path(__file__).resolve().parent

parser = build_arg_parser()
parser.set_defaults(
    pretrained_model_path=str(cwd / "pretrained_model"),
    finetuning_config=str(cwd / "finetune_config.json"),
    data_dir=str(cwd / "dataset"),
    datasetname="NaZrCl_5A_20N_Eform_400_fp64_size_100",
    modelname="finetune_foundation_model",
)
args = parser.parse_args()

os.environ["FINETUNING_LOG_DIR"] = str(cwd / "logs")
```

Keep `dictionary_variables` and the final `run_finetune(...)` call unchanged.
You can then override all settings directly:

```bash
# Finetune selected branches and convolution blocks.
python main.py \
  --selected-branches 5 7 \
  --unfreeze-conv-layers 0 1 \
  --num-epochs 50 \
  --batch-size 16 \
  --sample-interval 2 \
  --modelname nazrcl_branches_5_7

# Finetune every decoder branch while freezing the backbone.
python main.py \
  --all-branches \
  --freeze-backbone \
  --num-epochs 25 \
  --modelname nazrcl_all_heads
```

Useful CLI options include:

| Option | Purpose |
|---|---|
| `--pretrained-model-path PATH` | Directory containing the source `config.json` and checkpoints |
| `--finetuning-config PATH` | Finetuning JSON file |
| `--data-dir PATH` | Directory containing the pickled dataset |
| `--datasetname NAME` | Dataset filename stem, without `.pkl` |
| `--modelname NAME` | Name used for logs, configuration, and saved model files |
| `--checkpoint PATH` | Use a specific checkpoint |
| `--selected-branches N [N ...]` | Finetune selected zero-based decoder branches |
| `--all-branches` | Finetune all decoder branches |
| `--freeze-backbone` | Train selected decoders while freezing the backbone |
| `--unfreeze-conv-layers N [N ...]` | Freeze the backbone except for selected convolution blocks |
| `--batch-size N` | Override the JSON batch size |
| `--num-epochs N` | Override the JSON epoch count |
| `--sample-interval N` | Retain every Nth dataset sample |
| `--seed N` | Set the dataset split/random seed |

## 6. Outputs

When launched from this folder, HydraGNN writes results under:

```text
logs/<modelname>/
```

The output includes `run.log`, TensorBoard events, a saved configuration that
combines the pretrained and finetuning settings, and the final model
checkpoint. The console output also reports the checkpoint that was loaded,
the selected branches, dataset split sizes, available convolution blocks, and
the trainable-parameter summary.

## Common errors

- **Dataset path ends in `.pkl.pkl`**: pass `--datasetname` or set
  `args.datasetname` without the `.pkl` extension.
- **`No .pk checkpoint found`**: verify that `pretrained_model_path` points to
  the directory containing the downloaded checkpoint.
- **Branch is absent from `graph_shared` or `heads_NN`**: choose a branch that
  exists in the pretrained model. The reported available branch names can help
  identify valid indices.
- **Convolution index is out of range**: convolution blocks are zero-based; use
  one of the indices printed by the script.
- **A CLI option appears to be ignored**: check whether `main.py` assigns that
  value after `parse_args()`. Use the CLI-first pattern above.
- **Import error for `utils` or `hydragnn`**: set `PYTHONPATH` as shown in the
  environment setup section and run from the repository checkout.

## Running in a SLURM HPC
First install HydraGNN as described in the original repository under the folder installation_DOE_supercomputers.
Below is an example SLURM script used to finetune the pretrained model on Perlmutter

```bash

#!/bin/bash
#SBATCH -A m5216
#SBATCH --job-name=finetune
#SBATCH --output=hydra_%j.out
#SBATCH --error=hydra_%j.err
#SBATCH --time=00:30:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=omar_oraby@uml.edu
#SBATCH --mail-type=ALL

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/global/homes/o/oraby122/HydraGNN}
VENV_PATH=${VENV_PATH:-$HYDRAGNN_ROOT/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv}
CWD=$PWD
# --- Perlmutter module + conda setup ---
module purge
module reset
ml nersc-default/1.0 || true
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0
ml cudatoolkit/13.2


if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found."
    exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: VENV_PATH does not exist: $VENV_PATH"
    echo "Set VENV_PATH to your Perlmutter HydraGNN conda env path."
    exit 1
fi

conda activate "$VENV_PATH"

cd "$HYDRAGNN_ROOT" || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH

# Going back to original directory
cd "$CWD" || exit 1
echo "Current Working Directory: $CWD"

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'



export OMP_NUM_THREADS=8

echo "Tasks Per Node"
echo $SLURM_TASKS_PER_NODE
echo "Total Tasks"
echo $((SLURM_JOB_NUM_NODES*SLURM_TASKS_PER_NODE))
echo "-------------------------------"

echo "===== Submitted Script ====="
cat main.py
echo "============================"


cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c8 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -l --kill-on-bad-exit=1 \
	--export=ALL python -u main.py 

```

