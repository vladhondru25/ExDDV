#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/video_data.json \
    --image_folder /path/to/your/image/folder \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/test_train \
    --num_crops 4 \
    --max_num_frames 32 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --projector_lr 2e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to none \
    --lazy_preprocess True \
    --dataloader_num_workers 4