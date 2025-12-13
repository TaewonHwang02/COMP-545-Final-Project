#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --time=0-10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --job-name=bagel_mme_baseline
#SBATCH --output=logs_zero_ablation/mme_baseline.out
#SBATCH --error=logs_zero_ablation/mme_baseline.err

set -x

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

source /lustre06/project/6067888/thwang4/bagel/venv/bin/activate

# IMPORTANT: make sure ABLATE_LAYER is not set
unset ABLATE_LAYER

# Set paths
export model_path="/lustre06/project/6067888/thwang4/bagel/BAGEL/models/BAGEL-7B-MoT"
export output_path="/lustre06/project/6067888/thwang4/bagel/BAGEL/mme_baseline"

# OpenAI key if your script needs it (can be empty if not used)
export OPENAI_API_KEY=""

export GPUS=1
DATASETS=("mme")
DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

# No subsample → full MME
export SUBSAMPLE=0

bash scripts/eval/run_eval_vlm.sh
