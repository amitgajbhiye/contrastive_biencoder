#!/bin/bash --login

#SBATCH --job-name=ep4GS_BPM-BestPretrainedModelSearch

#SBATCH --output=logs/mcrae_fine_tune/out_grid1_ep4_search_best_model_mcrae_pcv_bb_cbien_cnetp.txt
#SBATCH --error=logs/mcrae_fine_tune/err_grid1_ep4_search_best_model_mcrae_pcv_bb_cbien_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --qos="gpu7d"
#SBATCH -t 7-00:00:00


conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/grid1_ep4_search_best_model_mcrae_pcv_bb_cbien_cnetp.json

echo 'Job Finished !!!'
