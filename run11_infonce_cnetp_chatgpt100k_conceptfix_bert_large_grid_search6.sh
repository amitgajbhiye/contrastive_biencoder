#!/bin/bash --login

#SBATCH --job-name=g6100kcChatGPT_pretrain

#SBATCH --output=logs/chatgpt_pretrain_grid_search/out_infonce_cnetp_chatgpt100k_conceptfix_bert_large_grid_search6.txt
#SBATCH --error=logs/chatgpt_pretrain_grid_search/err_infonce_cnetp_chatgpt100k_conceptfix_bert_large_grid_search6.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G

#SBATCH --time 7-00:00:00
#SBATCH --qos=gpu7d


conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/infonce_cnetp_chatgpt100k_conceptfix_bert_large_grid_search6.json

echo 'Job Finished!'
