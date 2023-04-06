#!/bin/bash --login

#SBATCH --job-name=1runConPropLoss

#SBATCH --output=logs/conprop_fix_infonce_loss_pretrain/out_one_run_conprop_fix_bienc_infonce_bert_base_cnetp_pretrain.txt
#SBATCH --error=logs/conprop_fix_infonce_loss_pretrain/err_one_run_conprop_fix_bienc_infonce_bert_base_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

#SBATCH --time 0-10:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/one_run_conprop_fix_bienc_infonce_bert_base_cnetp_pretrain.json

echo 'Job Finished!'
