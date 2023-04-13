#!/bin/bash --login

#SBATCH --job-name=GS_LargeBS

#SBATCH --output=logs/mcrae_fine_tune/out_grid_search_best_model_large_batch_size_mcrae_pcv_bb_cbien_cnetp_finetune.txt
#SBATCH --error=logs/mcrae_fine_tune/err_grid_search_best_model_large_batch_size_mcrae_pcv_bb_cbien_cnetp_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00


conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/grid_search_best_model_large_batch_size_mcrae_pcv_bb_cbien_cnetp_finetune.json

echo 'Job Finished !!!'
