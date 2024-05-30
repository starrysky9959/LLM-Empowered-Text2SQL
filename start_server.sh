#!/bin/bash
set -euxo pipefail
source .env

python -m vllm.entrypoints.openai.api_server \
    --served-model-name deepseek \
    --model=../local_models/deepseek-ai/deepseek-coder-6.7b-instruct/ \
    --gpu-memory-utilization=0.6 \
    --tensor-parallel-size=2 \
    --max-model-len=8192 \
    --enable-lora \
    --lora-modules sql-lora=${OUTPUT_MODEL_STEP_2_DIR} 
