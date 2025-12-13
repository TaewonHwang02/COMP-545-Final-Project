# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# Set proxy and API key
export OPENAI_API_KEY=$openai_api_key

export GPUS=1
#"mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp"
DATASETS=("mme" )
# DATASETS=("mmmu-val_cot")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path \
    --subsample ${SUBSAMPLE:-0}