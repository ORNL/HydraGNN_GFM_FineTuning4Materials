#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J wiggle150-bench
#SBATCH -o wiggle150-bench-%j.out
#SBATCH -e wiggle150-bench-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

# ---- Environment ----
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm624

export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

# ---- Runtime settings ----
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

# ---- Project paths ----
PROJDIR="$SLURM_SUBMIT_DIR"
export PYTHONPATH="${PROJDIR}/HydraGNN:${PROJDIR}:${PYTHONPATH}"

echo "Python: $(which python)"
echo "Project: ${PROJDIR}"
echo "Node: $(hostname)"
rocm-smi || true

# ---- Run benchmark (single GPU, small dataset) ----
# NOTE: Raw data and pickle must be prepared BEFORE submitting.
#       Run this on the login node first:
#
#         cd /path/to/HydraGNN_GFM_FineTuning4Materials
#         export PYTHONPATH="$PWD/HydraGNN:$PWD"
#         python examples/wiggle150/wiggle150_preonly.py
#
echo "=== Running benchmark ==="
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u "${PROJDIR}/examples/wiggle150/run_benchmark.py"

echo "=== Done ==="
