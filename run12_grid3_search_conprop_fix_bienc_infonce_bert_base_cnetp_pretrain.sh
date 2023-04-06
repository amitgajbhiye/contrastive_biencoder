#!/bin/bash --login

#SBATCH --job-name=G3ConPropFixLoss

#SBATCH --output=logs/conprop_fix_infonce_loss_pretrain/out_grid3_search_conprop_fix_bienc_infonce_bert_base_cnetp_pretrain.txt
#SBATCH --error=logs/conprop_fix_infonce_loss_pretrain/err_grid3_search_conprop_fix_bienc_infonce_bert_base_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=10G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/grid3_search_conprop_fix_bienc_infonce_bert_base_cnetp_pretrain.json

echo 'Job Finished!'
