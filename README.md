# LLM-Empowered-Text2SQL

## Model

[DeepSeek Coder Instruct](https://github.com/deepseek-ai/deepseek-coder/)

## Evaluation on Spider

```bash
python evaluation.py --gold ../spider/dev_gold.sql --pred ../LLM-Empowered-Text2SQL/result.txt --etype all --db ../spider/database/ --table ../spider/tables.json 
```
## 
```bash
sbatch -x paraai-n32-h-01-agent-[1-44],paraai-n32-h-01-agent-[48-56],paraai-n32-h-01-agent-[63-100] --gpus=1 ./run.sh 
```

## ClickHouse Datasource

["What's on the Menu?" dataset](https://clickhouse.com/docs/en/getting-started/example-datasets/menus)

## Finetune

deepspeed --num_gpus 2 finetune_deepseekcoder.py \
    --model_name_or_path /date1/luzhan/projects/deepseek-ai/deepseek-coder-6.7b-instruct \
    --data_path /date1/luzhan/projects/LLM-Empowered-Text2SQL/dataset.json \
    --output_dir ./output \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True 


# merge
```bash
python src/export_model.py \
    --model_name_or_path ../deepseek-coder-7b-instruct-v1.5 \
    --adapter_name_or_path ./output_2 \
    --template deepseekcoder \
    --finetuning_type lora \
    --export_dir ./merged_2 \
    --export_size 10 \
    --export_legacy_format False
```

```bash
python3 evaluation.py \
    --gold ../spider/dev_gold.sql \
    --pred ../LLM-Empowered-Text2SQL/public_dataset/bench/spider_dev_quant4bit.txt \
    --table ../spider/tables.json \
    --db ../spider/database/ \
    --etype all > 20240530_quant4bit.log
```

```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m
python -m vllm.entrypoints.openai.api_server \
    --served-model-name deepseek \
    --model /home/data2/luzhan/projects/LLM-Empowered-Text2SQL/finetuned_model/merged \
    --max-model-len 8196 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \
    --disable-log-requests
# --quantization awq \
    # --enable-lora \
    # --lora-modules sql-lora=/home/data2/luzhan/projects/LLM-Empowered-Text2SQL/finetuned_model/merged
```