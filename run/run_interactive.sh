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


#python3 ML_test.py
#python3 ML_test_wandb.py --epochs 2000 --lr 1e-5 --n-steps 10 --E_upper 1e-4 --E_lower 1e-6 \
	#-e 0.9 -a 1.0 --wandb-project AITimeStepper --wandb-name twobody_ML --optimizer lbfgs

#python ML_test_wandb.py --optimizer adam+lbfgs --adam-epochs 200 --epochs 400 --wandb-name test_run --lr 1e-4 --lbfgs-lr=1e-3


python ML_history_wandb.py --epochs 2000 --n-steps 10 --history-len 5 --feature-type delta_mag --save-name test_history_logs --wandb-project AITimeStepper --wandb-name history_ml
