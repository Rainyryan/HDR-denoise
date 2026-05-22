#!/bin/bash
#SBATCH --job-name=teacher_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=stanchan
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8      # <-- Increased from 4 to feed the bigger batches fast
#SBATCH --mem=240G              # <-- Scaled up memory headroom for batch data expansion
#SBATCH --time=6:00:00

# 1. Ensure directory structures exist
mkdir -p logs

# 2. Force Python to output terminal text immediately
export PYTHONUNBUFFERED=1

# 3. Securely pass your Weights & Biases API credentials to the automated worker node
# Replace the string below with the actual key from https://wandb.ai/authorize
export WANDB_API_KEY="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"

# 4. Set WANDB to online mode explicitly so it syncs up to the cloud dashboard
export WANDB_MODE=online

# 5. Direct Execution via the Absolute Path to your Environment Binary
/home/chen4848/.conda/envs/2025.06-py313/dl/bin/python train_HDR.py