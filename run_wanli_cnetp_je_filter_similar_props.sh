#!/bin/bash --login

#SBATCH --job-name=wanliCnet5negFilter

#SBATCH --output=logs/data_sampling/out_je_10neg_con_sim_clean_and_old_vocab_prop_filter_th0.50.txt
#SBATCH --error=logs/data_sampling/err_je_10neg_con_sim_clean_and_old_vocab_prop_filter_th0.50.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --time 0-03:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/je_wanli_pretrained_filter_sim_clean_props.json

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/je_wanli_pretrained_filter_sim_props.json


echo 'Job Finished!'
