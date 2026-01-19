#!/bin/bash

#module reset 
module purge
module load python  
module laod cuda/11.8
#module laod cuda/12.8
module load ffmpeg
module list  

source $HOME/pyenv/torch/bin/activate
python -V
which python


python run/runner.py train --epochs 2000 --n-steps 10 --history-len 5 --feature-type delta_mag --num-particles 4 --save-name test_history_logs --wandb-project AITimeStepper --wandb-name history_ml
