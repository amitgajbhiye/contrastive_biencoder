#!/bin/bash --login

#SBATCH --job-name=ft100InChatMcRae

#SBATCH --output=logs/chatgpt_mcrae_fine_tune/out_infonce_chatgpt100k_mcrae_pcv_bb_finetune.txt
#SBATCH --error=logs/chatgpt_mcrae_fine_tune/err_infonce_chatgpt100k_mcrae_pcv_bb_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

#SBATCH --time 0-08:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/infonce_chatgpt100k_mcrae_pcv_bb_finetune.json

echo 'finished!'