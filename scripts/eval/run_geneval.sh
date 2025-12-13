# scripts/eval/run_geneval.sh

set -x

GPUS=1

# Use env vars if set, otherwise default
: "${MASTER_PORT:=12345}"
: "${MASTER_PORT_EVAL:=$((MASTER_PORT+1000))}"

# generate images
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=$GPUS \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  ./eval/gen/gen_images_mp.py \
  --output_dir "$output_path/images" \
  --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
  --batch_size 1 \
  --num_images 4 \
  --resolution 1024 \
  --max_latent_size 64 \
  --model-path "$model_path"

# calculate score
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=$GPUS \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT_EVAL}" \
  ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
  "$output_path/images" \
  --outfile "$output_path/results.jsonl" \
  --model-path ./eval/gen/geneval/model

# summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py "$output_path/results.jsonl"
