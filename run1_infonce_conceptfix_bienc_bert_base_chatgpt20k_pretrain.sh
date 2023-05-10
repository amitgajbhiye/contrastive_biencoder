#!/bin/bash --login

#SBATCH --job-name=infoChat20KPretrain

#SBATCH --output=logs/chatgpt_pretrain/out_infonce_conceptfix_bienc_bert_base_chatgpt20k_pretrain.txt
#SBATCH --error=logs/chatgpt_pretrain/err_infonce_conceptfix_bienc_bert_base_chatgpt20k_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=8G

#SBATCH --time 0-08:00:00


conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/infonce_conceptfix_bienc_bert_base_chatgpt20k_pretrain.json

echo 'Job Finished!'
