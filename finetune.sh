#!/bin/bash

# 检查是否提供了两个参数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 model_path dataset_path output_path"
    exit 1
fi

# 打印参数
echo "model_path: $1"
echo "dataset_path: $2"
echo "output_path: $3"
model_path=$1
dataset_path=$2
output_path=$3

mkdir ${output}/step_1
mkdir ${output}/step_2

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --model_name_or_path ${model_path} \
    --template deepseekcoder \
    --dataset ${dataset_path}/step_1.json \
    --output_dir ${output_path}/step_2.json \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16
