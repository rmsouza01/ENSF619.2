#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=~/software/miniconda3/bin:$PATH
source activate tf2gpu

python /home/roberto.medeirosdeso/Transfer-Learning/transfer-learning-imagenet.py
conda deactivate
