#!/bin/bash
#SBATCH --account=def-sreddy
#SBATCH --time=0-10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --job-name=bagel_mmlu_baseline
#SBATCH --output=logs_mmlu/mmlu_baseline.out
#SBATCH --error=logs_mmlu/mmlu_baseline.err

echo ">>> Starting MMLU baseline (NO ablation)"

# IMPORTANT: no ABLATE_LAYER here
unset ABLATE_LAYER   # just to be sure

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2

source /lustre06/project/6067888/thwang4/bagel/venv/bin/activate

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_CACHE_DIR="/lustre06/project/6067888/thwang4/bagel/hf_mmlu"

cd /lustre06/project/6067888/thwang4/bagel/BAGEL

export MODEL_PATH="/lustre06/project/6067888/thwang4/bagel/BAGEL/models/BAGEL-7B-MoT"
export OUTPUT_DIR="/lustre06/project/6067888/thwang4/bagel/BAGEL/results_mmlu"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs_mmlu

SUBJECTS="high_school_mathematics,high_school_physics,college_mathematics,abstract_algebra,computer_security,formal_logic"
MAX_QUESTIONS=100

echo ">>> Model:        $MODEL_PATH"
echo ">>> Output dir:   $OUTPUT_DIR"
echo ">>> Subjects:     $SUBJECTS"
echo ">>> Max Q / subj: $MAX_QUESTIONS"
echo ">>> HF_CACHE_DIR: $HF_CACHE_DIR"

python mmlu_ablation.py \
  --model-path "$MODEL_PATH" \
  --subjects "$SUBJECTS" \
  --output-dir "$OUTPUT_DIR" \
  --max-questions "$MAX_QUESTIONS"
