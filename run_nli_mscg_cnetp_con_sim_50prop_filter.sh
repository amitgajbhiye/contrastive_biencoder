#!/bin/bash --login

#SBATCH --job-name=nliFilSim

#SBATCH --output=logs/data_sampling/out_nli_mscg_cnetp_con_sim_50prop_filter.txt
#SBATCH --error=logs/data_sampling/err_nli_mscg_cnetp_con_sim_50prop_filter.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition compute

#SBATCH --time 1-00:00:00

##SBATCH --mem=10g
##SBATCH --gres=gpu:1


echo 'This script is running on:' hostname


conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/nli_mscg_cnetp_con_sim_50prop_filter.json


echo 'Job Finished!'
