#!/bin/bash

accelerate launch \
    --main_process_port 30000 \
    --config_file etc/deepspeed/accelerate_config_nproc4.yaml \
    sft.py \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --load_in_8bit \
    --learning_rate 0.0001 \
    --model_max_length 8192 \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --save_total_limit 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --attn_implementation flash_attention_2
