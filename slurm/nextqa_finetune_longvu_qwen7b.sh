#!/bin/bash

#SBATCH --job-name=nextqa_finetune_longvu_qwen7b
#SBATCH --output=/mnt/meg/yvs/LongVU/logs/nextqa_finetune_longvu_qwen7b_%j.out
#SBATCH --error=/mnt/meg/yvs/LongVU/logs/nextqa_finetune_longvu_qwen7b_%j.err
#SBATCH --gpus=8

cd /mnt/meg/yvs/LongVU/

pip install -r requirements.txt

pip install --upgrade "jinja2>=3.1.0"

export NUM_GPUS=$SLURM_GPUS
echo "NUM_GPUS: $NUM_GPUS"
# Add environment variable to control memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

bash scripts/train_video_qwen.sh    