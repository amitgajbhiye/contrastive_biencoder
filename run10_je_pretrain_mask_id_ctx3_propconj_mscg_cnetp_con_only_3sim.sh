#!/bin/bash --login

#SBATCH --job-name=MLMctx3Co

#SBATCH --output=logs/pretrain/out_je_pretrain_mask_id_ctx3_propconj_mscg_cnetp_con_only_3sim.txt
#SBATCH --error=logs/pretrain/err_je_pretrain_mask_id_ctx3_propconj_mscg_cnetp_con_only_3sim.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=10g

#SBATCH --gres=gpu:1
#SBATCH --time 2-00:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/1_configs_pretrain/je_pretrain_mask_id_ctx3_propconj_mscg_cnetp_con_only_3sim.json

echo 'Job Finished!'
