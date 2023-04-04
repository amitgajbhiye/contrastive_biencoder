#!/bin/bash --login

#SBATCH --job-name=gsBestPretrainedModel

#SBATCH --output=logs/mcrae_fine_tune/out_grid_search_best_model_mcrae_pcv_bb_cbien_cnetp.txt
#SBATCH --error=logs/mcrae_fine_tune/err_grid_search_best_model_mcrae_pcv_bb_cbien_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-00:30:00


conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/grid_search_best_model_mcrae_pcv_bb_cbien_cnetp.json

echo 'Job Finished !!!'
