#!/bin/bash --login

#SBATCH --job-name=1NconPropCSLBPretrain

#SBATCH --output=logs/pretrain/out_deberta_v3_large_pretrain_mask_id_ctx1_cslb_1neg_con_prop.txt
#SBATCH --error=logs/pretrain/err_deberta_v3_large_pretrain_mask_id_ctx1_cslb_1neg_con_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=32g
#SBATCH --gres=gpu:1

#SBATCH --time 0-01:30:00

conda activate venv

python3 model/lm_con_prop.py --config_file configs/1_configs_pretrain/deberta_v3_large_pretrain_mask_id_ctx1_cslb_1neg_con_prop.json

echo 'Job Finished!'
