#!/bin/bash --login

#SBATCH --job-name=O_hp_infonce

#SBATCH --output=logs/pretrain/out_optuna_hp_bienc_infonce_bert_base_cnetp_pretrain.txt
#SBATCH --error=logs/pretrain/err_optuna_hp_bienc_infonce_bert_base_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

#SBATCH --time 0-2:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/optuna_hp_bienc_infonce_bert_base_cnetp_pretrain.json

echo 'Job Finished!'
