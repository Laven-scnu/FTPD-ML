#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-InternVL2_5}"
MODEL_TYPE="${MODEL_TYPE:-internvl2_5}"
DATASET_PATH="${DATASET_PATH:-data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/internvl_lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES

swift sft \
    --model "${MODEL_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --device_map auto \
    --train_type lora \
    --dataset "${DATASET_PATH}" \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir "${OUTPUT_DIR}" \
    --system "You are a helpful assistant." \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
