#!/bin/bash

#SBATCH --job-name=longvu_eval_nextqa
#SBATCH --output=/mnt/meg/yvs/LongVU/logs/eval_nextqa_%j.out
#SBATCH --error=/mnt/meg/yvs/LongVU/logs/eval_nextqa_%j.err
#SBATCH --gpus=1

cd /mnt/meg/yvs/LongVU

pip install -r requirements.txt

#echo "Checking transformers version:"
#pip3 show transformers | grep Version

#echo ""
#echo "Checking active conda environment:"
#conda info

export NUM_GPUS=$SLURM_GPUS
echo "NUM_GPUS: $NUM_GPUS"
# Add environment variable to control memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Add environment variable to enable work distribution across GPUs
#export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nnodes 1 --nproc_per_node $NUM_GPUS eval/eval_nextqa.py --data_path ./data/nextqa --version qwen --model_path checkpoints/longvu_cambrian_qwen