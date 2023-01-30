#!/bin/bash --login

#SBATCH --job-name=MLMctx3PPpretrain

#SBATCH --output=logs/pretrain/out_roberta_base_je_pretrain_mask_id_ctx3_propconj_mscg_cnetp_prdict_prop_3sim.txt
#SBATCH --error=logs/pretrain/err_roberta_base_je_pretrain_mask_id_ctx3_propconj_mscg_cnetp_prdict_prop_3sim.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=10g

#SBATCH --time 0-02:00:00
#SBATCH --gres=gpu:1

echo 'This script is running on:' hostname

conda activate venv

python3 model/roberta_prop_aug.py --config_file configs/1_configs_pretrain/roberta_base_je_pretrain_mask_id_ctx3_propconj_mscg_cnetp_prdict_prop_3sim.json

echo 'Job Finished!'
