#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2    # <- match to OMP_NUM_THREADS
###SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bgak-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=sim_gpu
#SBATCH --time=02:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
###SBATCH --array=0-2 

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
##SBATCH --mail-user=you@yourinstitution.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options


module reset 
module load python  
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

epss=(1 0.1 0.01)
eps=${epss[$SLURM_ARRAY_TASK_ID]}
#echo "Running array task $SLURM_ARRAY_TASK_ID with dt=$dt"

echo "job is starting on `hostname`"
#srun python3 run/runner.py simulate

#python3 run/runner.py simulate --dt "$dt" --steps "$step"
#python3 run/runner.py simulate --eps "$eps" --steps 200
#python3 run/runner.py simulate --integrator-mode ml --model-path "$MODEL_PATH"

#MODEL_PATH="/u/gkerex/projects/AITimeStepper/data/test_history_logs/model/model_epoch_0250.pt"
#python run/runner.py simulate --integrator-mode history --model-path=$MODEL_PATH --history-len 5 --num-particles 4 --steps 500 --save-name="history_test_dL"

#MODEL_PATH="/u/gkerex/projects/AITimeStepper/data/test_history_logs/model/model_epoch_0200.pt"
#python run/runner.py simulate --integrator-mode history --model-path=$MODEL_PATH --history-len 20 --num-particles 4 --steps 500 --save-name="history_test_dL_20_1"
MODEL_PATH="/u/gkerex/projects/AITimeStepper/data/test_history_multi_logs/model/model_epoch_0100.pt"
python run/runner.py simulate --integrator-mode history --model-path=$MODEL_PATH --history-len 5 --num-particles 8 --steps 500 --save-name="history_test_dL_5_multi"
