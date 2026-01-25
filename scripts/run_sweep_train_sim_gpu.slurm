#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6    # <- match to OMP_NUM_THREADS
###SBATCH --partition=gpuA40x4  # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bgak-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=aiT_sweep
#SBATCH --time=12:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
#SBATCH --array=0-3

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
##SBATCH --mail-user=you@yourinstitution.edu
##SBATCH --mail-type="BEGIN,END"

module reset
module load python
module load ffmpeg
module list

source $HOME/pyenv/torch/bin/activate
python -V
which python

echo "job is starting on $(hostname)"

CONFIGS=(
  "configs/sweep/sweep_01.toml"
  #"configs/sweep/sweep_02.toml"
  #"configs/sweep/sweep_03.toml"
  #"configs/sweep/sweep_04.toml"
)

CONFIG_PATH=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running config: ${CONFIG_PATH}"
python run/runner.py both --config "${CONFIG_PATH}"
