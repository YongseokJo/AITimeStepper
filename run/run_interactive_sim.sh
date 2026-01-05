#!/bin/bash

module reset 
module load python  
module laod cuda/11.8.0
module load ffmpeg
module list  

source $HOME/pyenv/torch/bin/activate
python -V
which python


#python3 simulator_test.py --e 0.9 --ML
#python3 simulator_test.py --e=0.9 --a=1 --ML
#python3 simulator_test.py --e=0.3 --a=1 --ML
#python3 simulator_test.py --e=0.9 --a=0.5 --ML

MODEL_PATH="/u/gkerex/projects/AITimeStepper/data/test_history_logs/model/model_epoch_1000.pt"


python simulator_test.py --ML --model-path=$MODEL_PATH --history-len 5 --a=1 --e=0.9 --save-name="history_test"

