#!/bin/bash
#SBATCH --job-name=mmlu
#SBATCH --account=ctb-timod 
#SBATCH --time=32:00:00 
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --output=logs/bagel_%j.out
#SBATCH --error=logs/bagel_%j.err 
               
source venv/bin/activate

python bagel_reasoning_mmlu.py 2>&1 | tee mmlu_bagel_logs_server.txt