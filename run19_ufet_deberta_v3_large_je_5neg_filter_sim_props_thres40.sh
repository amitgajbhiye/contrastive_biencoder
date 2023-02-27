#!/bin/bash --login

#SBATCH --job-name=filterUfetCon50Sim 

#SBATCH --output=logs/ufet_exp/out_ufet_zero_shot_deberta_v3_large_je_5neg_filter_sim_props_thres50.txt
#SBATCH --error=logs/ufet_exp/err_ufet_zero_shot_deberta_v3_large_je_5neg_filter_sim_props_thres50.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=0
#SBATCH -t 0-09:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_zero_shot_deberta_v3_large_je_5neg_filter_sim_props_thres50.json

echo 'Job Finished!'