#!/bin/bash
set -euxo pipefail
source .env

# 打印参数
echo "USER_DATASET_PATH: ${USER_DATASET_PATH}"
echo "FINETUNE_DATASET_DIR: ${FINETUNE_DATASET_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "FINETUNE_DATASET_STEP_1_PATH:${FINETUNE_DATASET_STEP_1_PATH}"
echo "FINETUNE_DATASET_STEP_2_PATH:${FINETUNE_DATASET_STEP_2_PATH}"

#!/bin/bash

# 定义要操作的文件夹列表
folders=("${FINETUNE_DATASET_DIR}" "${OUTPUT_MODEL_DIR}" "${OUTPUT_MODEL_STEP_1_DIR}" "${OUTPUT_MODEL_STEP_1_DIR}" "${EXPORT_MODEL_DIR}")

# 遍历文件夹列表
for folder in "${folders[@]}"; do
    # 检查文件夹是否存在
    if [ ! -d "$folder" ]; then
        # 文件夹不存在，创建文件夹
        mkdir "$folder"
        echo "Folder $folder created."
    else
        echo "Folder $folder already exists."
    fi
done

python3 build_finetune_dataset.py
exit 0

cd LLaMA-Factory/
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../text2sql_step_1_lora_sft.yaml
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train ../text2sql_step_2_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
nohup CUDA_VISIBLE_DEVICES=1 llamafactory-cli train ../text2sql_lora_sft.yaml > ../log/logfile.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0,1
nohup accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py ../text2sql_lora_sft.yaml > ../log/logfile.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

# CUDA_VISIBLE_DEVICES=0 python LLaMA-Factory/src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --finetuning_type lora \
#     --lora_target all \
#     --lora_rank 8 \
#     --lora_alpha 16 \
#     --model_name_or_path ${MODEL_PATH} \
#     --template ${MODEL_TEMPLATE} \
#     --dataset ${FINETUNE_DATASET_STEP_1_PATH} \
#     --output_dir ${OUTPUT_MODEL_STEP_1_DIR} \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --warmup_steps 10 \
#     --save_steps 100 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --bf16
