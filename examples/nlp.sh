#!/bin/bash -l

#SBATCH --job-name="nlp"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=32G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate HuggingfaceActiveLearning
export CUDA_VISIBLE_DEVICES=1

python ./examples/nlp.py \
--learn \
--evaluate \
--dataset="ag_news" \
--pretrained_model_name_or_path="distilbert-base-uncased" \
--metric="accuracy" \
--querier=$1 \
--stopper="null" \
--n_iterations=32 \
--log_level=50 \
--n_start=32 \
--n_query=32
