#!/bin/bash --login

#SBATCH --job-name=GSconPropFix_BestPretrained_ModelSearch

#SBATCH --output=logs/mcrae_fine_tune/out_grid_search_conpropfix_mcrae_pcv_bb_cbien_cnetp_pretrained_model.txt
#SBATCH --error=logs/mcrae_fine_tune/err_grid_search_conpropfix_mcrae_pcv_bb_cbien_cnetp_pretrained_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH --mem=10g

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1
##SBATCH --qos="gpu7d"

#SBATCH -t 0-01:00:00


conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/grid_search_conpropfix_mcrae_pcv_bb_cbien_cnetp_pretrained_model.json

echo 'Job Finished !!!'
