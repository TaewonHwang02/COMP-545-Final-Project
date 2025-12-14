#!/bin/bash
#SBATCH --job-name=rise_ablation           # Job name
#SBATCH --account=<YOUR_ACCOUNT>           # Your compute allocation account
#SBATCH --time=20:00:00                    # Max runtime (20 hours)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # CPUs per task
#SBATCH --mem=128G                         # Memory (RISE generation needs more)
#SBATCH --gres=gpu:a100:2                  # GPU resources (2 x A100)
#SBATCH --output=logs/rise_ablation_%j.out # Standard output file
#SBATCH --error=logs/rise_ablation_%j.err  # Standard error file
#SBATCH --mail-type=ALL                    # Mail notification type
#SBATCH --mail-user=<YOUR_EMAIL>           # Your email address

echo "================================================================================"
echo "RISE Ablation Evaluation - With/Without ViT Comparison Experiment"
echo "================================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "================================================================================"

# Create log directory
mkdir -p logs

# Load modules (adjust according to your cluster configuration)
module load StdEnv/2023 python/3.10.13 opencv/4.10.0 cuda/12.2

# Activate virtual environment (adjust path to your environment)
source /path/to/your/virtualenv/bin/activate

# Disable torch compile and dynamo for stability
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Set OpenAI API Key (for GPT evaluation)
export OPENAI_API_KEY='<YOUR_OPENAI_API_KEY>'

# ============================================================================
# Configuration Parameters
# ============================================================================

MODEL_PATH="/path/to/checkpoints/BAGEL-7B-MoT"
METADATA_FILE="./eval/gen/rise/data/datav2_total_w_subtask.json"
RISE_DATA_DIR="./eval/gen/rise/data"
OUTPUT_DIR="./eval_results/rise_ablation_${SLURM_JOB_ID}"

# Sample count estimation:
# - 12 hours = 43200 seconds
# - Model loading: ~120 seconds
# - Each sample generates 2 images (no ViT + with ViT)
# - Each image ~50 timesteps, ~2 seconds per timestep = 100 seconds/image
# - Total time per sample: 200 seconds
# - Available samples: (43200 - 240) / 200 = ~215
# - Conservative estimate: 150 samples
NUM_SAMPLES=360

# GPU Configuration (2 x A100)
# Model will be automatically distributed across 2 GPUs
MAX_MEM_PER_GPU="40GiB"  # Max memory per GPU

# Inference parameters
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=1.5
NUM_TIMESTEPS=50

echo ""
echo "================================================================================"
echo "Configuration Info"
echo "================================================================================"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Metadata file: $METADATA_FILE"
echo "Number of test samples: $NUM_SAMPLES"
echo ""
echo "GPU Configuration:"
echo "  - GPU count: 2 x A100"
echo "  - Memory per GPU: $MAX_MEM_PER_GPU"
echo "  - Model will be auto-distributed across GPUs"
echo ""
echo "Inference parameters:"
echo "  - CFG Text Scale: $CFG_TEXT_SCALE"
echo "  - CFG Image Scale: $CFG_IMG_SCALE"
echo "  - Timesteps: $NUM_TIMESTEPS"
echo ""
echo "Estimated time: ~8-10 hours (including image generation)"
echo "================================================================================"
echo ""

# Display GPU info
echo "Detected GPUs:"
nvidia-smi --list-gpus
echo ""

# ============================================================================
# Check Required Files and Directories
# ============================================================================

# Change to project directory
cd "$(dirname "$0")"

echo "Checking dependencies..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/ema.safetensors" ]; then
    echo "Error: Model weights not found: $MODEL_PATH/ema.safetensors"
    exit 1
fi

if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: RISE metadata file does not exist: $METADATA_FILE"
    echo "Please download RISE data following instructions in EVAL.md"
    exit 1
fi

if [ ! -d "$RISE_DATA_DIR/data" ]; then
    echo "Error: RISE data directory does not exist: $RISE_DATA_DIR/data"
    exit 1
fi

echo "All checks passed"
echo ""

# ============================================================================
# Phase 1/2: Image Generation (With/Without ViT Comparison)
# ============================================================================

echo "================================================================================"
echo "Phase 1/2: Image Generation - Ablation Comparison"
echo "================================================================================"
echo ""
echo "Experiment setup:"
echo "  - Each sample generates 2 images"
echo "    - No ViT version: VAE tokens only"
echo "    - With ViT version: VAE + ViT tokens"
echo ""
echo "Estimated time: ~8 hours"
echo "Output location: $OUTPUT_DIR"
echo ""
echo "Starting generation... ($(date))"
echo ""

python run_rise_ablation_eval.py \
    --model-path $MODEL_PATH \
    --metadata-file $METADATA_FILE \
    --rise-data-dir $RISE_DATA_DIR \
    --output-dir ./eval_results \
    --num-samples $NUM_SAMPLES \
    --cfg-text-scale $CFG_TEXT_SCALE \
    --cfg-img-scale $CFG_IMG_SCALE \
    --num-timesteps $NUM_TIMESTEPS \
    --max-mem-per-gpu $MAX_MEM_PER_GPU \
    --seed 42

EXIT_CODE=$?

echo ""
echo "Image generation completed ($(date))"
echo ""

if [ $EXIT_CODE -ne 0 ]; then
    echo "Image generation failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Phase 1 completed - Images generated"
echo ""

# ============================================================================
# Find Latest Result Directory
# ============================================================================

LATEST_RESULT_DIR=$(ls -td ./eval_results/rise_ablation_* | head -1)
echo "Latest result directory: $LATEST_RESULT_DIR"

if [ ! -d "$LATEST_RESULT_DIR" ]; then
    echo "Error: Result directory not found"
    exit 1
fi

NO_VIT_DIR="$LATEST_RESULT_DIR/no_vit"
WITH_VIT_DIR="$LATEST_RESULT_DIR/with_vit"

# Count generated images
NO_VIT_COUNT=$(ls -1 $NO_VIT_DIR/*.png 2>/dev/null | wc -l)
WITH_VIT_COUNT=$(ls -1 $WITH_VIT_DIR/*.png 2>/dev/null | wc -l)

echo ""
echo "Generated image statistics:"
echo "  - No ViT version: $NO_VIT_COUNT images"
echo "  - With ViT version: $WITH_VIT_COUNT images"
echo ""

# ============================================================================
# Phase 2/2: GPT Evaluation (Optional)
# ============================================================================

echo "================================================================================"
echo "Phase 2/2: GPT Evaluation"
echo "================================================================================"
echo ""
echo "Evaluation setup:"
echo "  - Model: GPT-4 (via OpenAI API)"
echo "  - Metrics: Multi-dimensional automatic scoring"
echo ""
echo "Estimated time: Depends on sample count and API speed"
echo "Note: This will call OpenAI API and incur costs"
echo ""

# Evaluate no ViT version
echo "Evaluating no ViT version..."
python ./eval/gen/rise/gpt_eval.py \
    --data $METADATA_FILE \
    --input $RISE_DATA_DIR \
    --output $NO_VIT_DIR

if [ $? -eq 0 ]; then
    echo "No ViT version evaluation completed"
else
    echo "Warning: No ViT version evaluation failed (possibly API issue)"
fi

echo ""

# Evaluate with ViT version
echo "Evaluating with ViT version..."
python ./eval/gen/rise/gpt_eval.py \
    --data $METADATA_FILE \
    --input $RISE_DATA_DIR \
    --output $WITH_VIT_DIR

if [ $? -eq 0 ]; then
    echo "With ViT version evaluation completed"
else
    echo "Warning: With ViT version evaluation failed (possibly API issue)"
fi

echo ""
echo "Phase 2 completed"
echo ""

# ============================================================================
# Generate Comparison Report
# ============================================================================

echo "================================================================================"
echo "Generating Comparison Analysis Report"
echo "================================================================================"
echo ""

# Create comparison script
cat > $LATEST_RESULT_DIR/compare_results.py << 'EOF'
#!/usr/bin/env python3
"""Compare GPT evaluation results between no-ViT and with-ViT versions."""

import json
import pickle
from pathlib import Path


def load_results(result_dir):
    """Load GPT evaluation results."""
    pkl_file = Path(result_dir) / "judge_results.pkl"
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    return None


def compare_results(no_vit_dir, with_vit_dir):
    """Compare two sets of results."""
    no_vit_results = load_results(no_vit_dir)
    with_vit_results = load_results(with_vit_dir)
    
    print("=" * 80)
    print("RISE Ablation Comparison Analysis")
    print("=" * 80)
    
    if no_vit_results and with_vit_results:
        print("\nBoth evaluation results loaded successfully")
        print(f"  - No ViT samples: {len(no_vit_results)}")
        print(f"  - With ViT samples: {len(with_vit_results)}")
        
        print("\nFor detailed comparison analysis, check:")
        print(f"  - No ViT results: {no_vit_dir}")
        print(f"  - With ViT results: {with_vit_dir}")
    else:
        print("\nWarning: Evaluation result files not found or incomplete")
        print("Please check if GPT evaluation ran successfully")
    
    print("")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: compare_results.py <no_vit_dir> <with_vit_dir>")
        sys.exit(1)
    
    compare_results(sys.argv[1], sys.argv[2])
EOF

chmod +x $LATEST_RESULT_DIR/compare_results.py

# Run comparison
python $LATEST_RESULT_DIR/compare_results.py $NO_VIT_DIR $WITH_VIT_DIR

# ============================================================================
# Results Summary
# ============================================================================

echo ""
echo "================================================================================"
echo "Experiment Completion Summary"
echo "================================================================================"
echo ""
echo "RISE Ablation evaluation completed!"
echo ""
echo "Result locations:"
echo "  - Main directory: $LATEST_RESULT_DIR"
echo "  - No ViT results: $NO_VIT_DIR"
echo "  - With ViT results: $WITH_VIT_DIR"
echo "  - Results log: $LATEST_RESULT_DIR/results.json"
echo ""
echo "Generated files:"
echo "  - No ViT images: $NO_VIT_COUNT"
echo "  - With ViT images: $WITH_VIT_COUNT"
if [ -f "$NO_VIT_DIR/judge_results.pkl" ]; then
    echo "  - No ViT evaluation: $NO_VIT_DIR/judge_results.pkl"
fi
if [ -f "$WITH_VIT_DIR/judge_results.pkl" ]; then
    echo "  - With ViT evaluation: $WITH_VIT_DIR/judge_results.pkl"
fi
echo ""
echo "Follow-up analysis:"
echo "  1. Compare visual quality between two sets of images"
echo "  2. Analyze GPT score differences"
echo "  3. Quantify ViT tokens contribution"
echo ""
echo "Visualization command:"
echo "  # View specific samples"
echo "  eog $NO_VIT_DIR/*.png $WITH_VIT_DIR/*.png"
echo ""
echo "================================================================================"
echo "Job completed at: $(date)"
echo "================================================================================"
echo ""

# Resource usage statistics
echo "To view resource usage, run:"
echo "  seff $SLURM_JOB_ID"
echo ""

exit 0
