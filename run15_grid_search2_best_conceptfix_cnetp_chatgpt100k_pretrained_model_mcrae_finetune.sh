#!/bin/bash --login

#SBATCH --job-name=GSbmConFix

#SBATCH --output=logs/chatgpt_mcrae_fine_tune/out_grid_search2_best_conceptfix_cnetp_chatgpt100k_pretrained_model_mcrae_finetune.txt
#SBATCH --error=logs/chatgpt_mcrae_fine_tune/err_grid_search2_best_conceptfix_cnetp_chatgpt100k_pretrained_model_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

##SBATCH --qos=gpu7d

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/grid_search2_best_conceptfix_cnetp_chatgpt100k_pretrained_model_mcrae_finetune.json

echo 'Job Finished !!!'
