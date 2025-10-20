#!/bin/bash
#SBATCH -J poker             # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 720                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=5G
#SBATCH -q coc-ice
#SBATCH -o../logs/sft/Report-%j.out        # Combined output and error messages
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail preferences
#SBATCH --mail-user=shubohan@gatech.edu  # E-mail address for notifications

cd /home/hice1/bshu30/CS6220/sft

# Load modules
module load anaconda3

# Activate conda environment
conda activate cs6220

# Set HuggingFace cache to home directory (avoid permission issues)
export HF_HOME=/home/hice1/bshu30/.cache/huggingface
export TRANSFORMERS_CACHE=/home/hice1/bshu30/.cache/huggingface/transformers
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE

# Create necessary directories
mkdir -p ../logs/sft
mkdir -p ../poker-lora-model

# Show environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
nvidia-smi

python3 train_poker_model.py