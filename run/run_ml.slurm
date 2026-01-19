#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
###SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bgak-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=aiT_speedup
#SBATCH --time=12:00:00      # hh:mm:ss for the job

#SBATCH --constraint="scratch"
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --mail-type="BEGIN,END"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
###SBATCH --array=0-2 

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest


module reset 
module load python  
module laod cuda/11.8.0
module load ffmpeg
module list  

source $HOME/pyenv/torch/bin/activate
python -V
which python

#dts=(1 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4)
#dts=(0.3 0.7 0.03 0.07 5e-5 1e-5 5e-6 1e-6)
#dts=(1e-5 5e-6 1e-6)
#dt=${dts[$SLURM_ARRAY_TASK_ID]}


#steps=(10 100 1000 10000)
#step=${steps[$SLURM_ARRAY_TASK_ID]}

#echo "Running array task $SLURM_ARRAY_TASK_ID with dt=$dt"

echo "job is starting on `hostname`"
#srun python3 simulator_test.py

#python3 ML_test.py

#python ML_history_wandb.py --epochs 2000 --n-steps 20 --history-len 20 --feature-type delta_mag --save-name test_history_logs --wandb-project AITimeStepper --wandb-name history_ml_mag_20



python ML_history_multi_wandb.py --epochs 2000 --num-orbits 8 --debug --n-steps 5 --history-len 5 --feature-type delta_mag --save-name test_history_multi_logs --wandb-project AITimeStepper --wandb-name history_ml_multi_5
