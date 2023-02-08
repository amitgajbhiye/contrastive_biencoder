#!/bin/bash --login

#SBATCH --job-name=blbbMcPs

#SBATCH --output=logs/mcrae_finetune/out_bl_ps_berta_base_je_pretrain_mask_id_ctx1_con_prop_cnetp.txt
#SBATCH --error=logs/mcrae_finetune/err_bl_ps_berta_base_je_pretrain_mask_id_ctx1_con_prop_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-05:00:00


conda activate venv

python3 model/lm_con_prop.py --config_file configs/3_finetune/bl_ps_berta_base_je_pretrain_mask_id_ctx1_con_prop_cnetp.json

echo 'Job Finished!'