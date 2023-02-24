#!/bin/bash --login

#SBATCH --job-name=filterUfetCon50Sim 

#SBATCH --output=logs/ufet_exp/out_ufet_deberta_v3_large_je_5neg_filter_sim_props_thres40.txt
#SBATCH --error=logs/ufet_exp/err_ufet_deberta_v3_large_je_5neg_filter_sim_props_thres40.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH --mem=42G
#SBATCH -t 2-00:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_deberta_v3_large_je_5neg_filter_sim_props_thres40.json

echo 'Job Finished!'