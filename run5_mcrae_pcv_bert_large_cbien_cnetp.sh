#!/bin/bash --login

#SBATCH --job-name=MRblFT

#SBATCH --output=logs/mcrae_fine_tune/out_mcrae_pcv_bert_large_cbien_cnetp.txt
#SBATCH --error=logs/mcrae_fine_tune/err_mcrae_pcv_bert_large_cbien_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-08:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/mcrae_pcv_bert_large_cbien_cnetp.json

echo 'Job Finished !!!'
