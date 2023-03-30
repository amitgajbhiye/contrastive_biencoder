#!/bin/bash --login

#SBATCH --job-name=G2_CBien

#SBATCH --output=logs/pretrain/out_grid_search2_hp_bienc_infonce_bert_base_cnetp_pretrain.txt
#SBATCH --error=logs/pretrain/err_grid_search2_hp_bienc_infonce_bert_base_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition highmem

#SBATCH --time 0-01:00:00

##SBATCH --gres=gpu:1
##SBATCH --qos="gpu7d"

#SBATCH --mem=10G



conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/grid_search2_hp_bienc_infonce_bert_base_cnetp_pretrain.json

echo 'Job Finished!'
