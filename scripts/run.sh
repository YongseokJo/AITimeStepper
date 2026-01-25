#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --mail-type=ALL
#SBATCH --mail-user=g.kerex@gmail.com
#SBATCH --time=120:00:00
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu --gpus=1 -C a100

module add python3

~/pyenv/torch/bin/python3 main.py > stdout 2> stderr

