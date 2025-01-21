#!/bin/bash
#SBATCH --job-name=longvu_eval_mvbench
#SBATCH --output=/mnt/meg/yvs/LongVU/logs/eval_mvbench_%j.out
#SBATCH --error=/mnt/meg/yvs/LongVU/logs/eval_mvbench_%j.err
#SBATCH --gpus=1

cd /mnt/meg/yvs/LongVU

#pip install -r requirements.txt

#echo "Checking transformers version:"
#pip3 show transformers | grep Version

#echo ""
#echo "Checking active conda environment:"
#conda info

num_gpus=1
torchrun --standalone --nnodes 1 --nproc_per_node $num_gpus eval/eval_mvbench.py --data_path ./data/MVBench --version qwen --model_path checkpoints/longvu_qwen
#torchrun --nproc-per-node=8 eval/eval_mvbench.py --data_path ./data/MVBench --version qwen --model_path ./checkpoints/longvu_qwen
#python3 ./eval/eval_mvbench.py --data_path ./data/MVBench --version qwen --model_path ./checkpoints/longvu_qwen
