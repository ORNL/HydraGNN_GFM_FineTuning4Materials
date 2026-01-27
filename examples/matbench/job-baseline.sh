#!/bin/bash
#SBATCH -A LRN075
#SBATCH -J HydraGNN
#SBATCH -o job-base-%j.out
#SBATCH -e job-base-%j.out
#SBATCH -t 00:20:00
#SBATCH -p batch 
##SBATCH -q debug
#SBATCH -N 1 #16 
##SBATCH -S 1

 
# Load conda environemnt
module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/6.2.4
ml amd-mixed/6.2.4
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml cmake/3.27.9
#ml miniforge3/23.11.0
module unload darshan-runtime
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
conda activate /lustre/orion/stf006/proj-shared/irl1/HydraGNN_reinstall_again/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

#export PYTHONPATH=/lustre/orion/stf006/proj-shared/irl1/HydraGNN-main:/lustre/orion/stf006/proj-shared/irl1/H_GFM_FT4M:$PYTHONPATH
export PYTHONPATH=/lustre/orion/stf006/proj-shared/irl1/HydraGNN_reinstall_again:/lustre/orion/stf006/proj-shared/irl1/H_GFM_FT4M:$PYTHONPATH

which python
python -c "import numpy; print(numpy.__version__)"


echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1

for i in {0..4}; do
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --datasetname matbench_dielectric_"$i" --modelname matbench_dielectric_"$i"
  srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --datasetname matbench_jdft2d_"$i" --modelname matbench_jdft2d_"$i"
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --modelname matbench_log_gvrh_"$i"
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --modelname matbench_log_kvrh_"$i"
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --modelname matbench_mp_e_form_"$i" #Long change to 10 epoch
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --modelname matbench_mp_gap_"$i" #Long change to 10 epoch
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --modelname matbench_perovskites_"$i"
  #srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u examples/matbench/ensemble_fine_tune.py --modelname matbench_phonons_"$i"
done
