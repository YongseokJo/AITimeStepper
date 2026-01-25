#!/bin/bash

module reset 
module load python  
module laod cuda/11.8.0
module load ffmpeg
module list  

source $HOME/pyenv/torch/bin/activate
python -V
which python


#python3 run/runner.py simulate --integrator-mode ml --model-path "$MODEL_PATH"

MODEL_PATH="/u/gkerex/projects/AITimeStepper/data/test_history_logs/model/model_epoch_1000.pt"


python run/runner.py simulate --integrator-mode history --model-path=$MODEL_PATH --history-len 5 --num-particles 4 --steps 500 --save-name="history_test"
