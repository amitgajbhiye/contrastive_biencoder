#!/bin/bash --login

#SBATCH --job-name=dbV3Filter

#SBATCH --output=logs/ufet_exp/out_ufet_cslb_deberta_v3_large_filter_sim_props_thres50.txt
#SBATCH --error=logs/ufet_exp/err_ufet_cslb_deberta_v3_large_filter_sim_props_thres50.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --mem=18G
#SBATCH --gres=gpu:1

#SBATCH --time 0-01:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_cslb_deberta_v3_large_filter_sim_props_thres50.json

echo 'Job Finished!'
