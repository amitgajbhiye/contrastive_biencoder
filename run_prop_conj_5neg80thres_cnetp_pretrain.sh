#!/bin/bash --login

#SBATCH --job-name=5neg80thres

#SBATCH --output=logs/pretrain/out_je_propconj_5neg80thres_cnetp_pretrain.txt
#SBATCH --error=logs/pretrain/err_je_propconj_5neg80thres_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858
#SBATCH --partition gpu_v100,gpu

#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

##SBATCH --qos="gpu7d"

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/1_configs_pretrain/je_propconj_5neg80thres_cnetp_pretrain.json

echo 'Job Finished!'