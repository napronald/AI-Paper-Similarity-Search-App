#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=GPU
#SBATCH --output=results.txt

python3 main.py