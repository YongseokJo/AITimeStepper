#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bgak-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=sim
#SBATCH --time=02:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
###SBATCH --array=0-2 

### GPU options ###
##SBATCH --gpus-per-node=2
##SBATCH --gpu-bind=none     # <- or closest
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
#srun python3 simulator_test.py

#python3 simulator_test.py --dt "$dt" --e 0.9 --steps 
#python3 simulator_test.py --e 0.9 --steps "$step"
#python3 simulator_test.py --e 0.9 --eps "$eps" --Nperiod 10
#python3 simulator_test.py --e 0.7 --ML
#python3 simulator_test.py --e 0.9 --ML


MODEL_PATH="/u/gkerex/projects/AITimeStepper/data/test_history_logs/model/model_epoch_0250.pt"

python simulator_test.py --ML --model-path=$MODEL_PATH --history-len 5 --a=1 --e=0.9 --save-name="history_test_dL"

