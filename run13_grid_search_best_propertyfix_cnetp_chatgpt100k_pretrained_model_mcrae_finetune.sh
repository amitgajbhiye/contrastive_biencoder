#!/bin/bash --login

#SBATCH --job-name=GSbmPropFix

#SBATCH --output=logs/chatgpt_mcrae_fine_tune/out_grid_search_best_all_propertyfix_cnetp_chatgpt100k_pretrained_model_mcrae_finetune.txt
#SBATCH --error=logs/chatgpt_mcrae_fine_tune/err_grid_search_best_all_propertyfix_cnetp_chatgpt100k_pretrained_model_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos="gpu7d"

#SBATCH -t 7-00:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/grid_search_best_propertyfix_cnetp_chatgpt100k_pretrained_model_mcrae_finetune.json

echo 'Job Finished !!!'