#!/bin/bash --login

#SBATCH --job-name=rbC2DHPpt

#SBATCH --output=logs/pretrain/out_ctx2_roberta_base_je_pretrain_mask_id_con_prop_cnetp_default_hyperparam.txt
#SBATCH --error=logs/pretrain/err_ctx2_roberta_base_je_pretrain_mask_id_con_prop_cnetp_default_hyperparam.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-03:00:00

conda activate venv

python3 model/lm_con_prop.py --config_file configs/1_configs_pretrain/ctx2_roberta_base_je_pretrain_mask_id_con_prop_cnetp_default_hyperparam.json

echo 'Job Finished!'
