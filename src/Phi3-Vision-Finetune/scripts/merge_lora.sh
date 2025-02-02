#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/lora_vision_test \
    --model-base $MODEL_NAME  \
    --save-model-path /home/eivor/biodeep/xAI_deepfake/src/Phi3-Vision-Finetune/output/model_merged \
    --safe-serialization