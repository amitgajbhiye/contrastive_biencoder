#!/bin/bash --login

#SBATCH --job-name=G3JointLoss

#SBATCH --output=logs/entropy_infonce_joint_loss/out_grid3_search_jointloss_bienc_bert_base_cnetp_pretrain.txt
#SBATCH --error=logs/entropy_infonce_joint_loss/err_grid3_search_jointloss_bienc_bert_base_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/grid3_search_jointloss_bienc_bert_base_cnetp_pretrain.json

echo 'Job Finished!'
