#!/bin/bash
#SBATCH --account=def-sreddy             
#SBATCH --time=0-10:00                   
#SBATCH --gres=gpu:1                     
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-27%4                   
#SBATCH --job-name=bagel_layer_ablate
#SBATCH --output=logs_zero_ablation/layer_%a.out
#SBATCH --error=logs_zero_ablation/layer_%a.err

echo ">>> Starting ablation for layer $SLURM_ARRAY_TASK_ID"

# Set which layer to ablate
export ABLATE_LAYER=$SLURM_ARRAY_TASK_ID

# Modules
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

# Activate your venv
source /lustre06/project/6067888/thwang4/bagel/venv/bin/activate

# Python path
export PYTHONPATH=/lustre06/project/6067888/thwang4/bagel/BAGEL:$PYTHONPATH

# Output directory for each layer
export output_path="/lustre06/project/6067888/thwang4/bagel/BAGEL/zero_ablation_${ABLATE_LAYER}"
export model_path="/lustre06/project/6067888/thwang4/bagel/BAGEL/models/BAGEL-7B-MoT"
unset SUBSAMPLE

bash scripts/eval/run_eval_vlm.sh \
    "$OUT_DIR" \
  --model-path "$MODEL_PATH"
