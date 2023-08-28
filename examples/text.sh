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

python ./examples/main.py \
--task="text" \
--learn \
--evaluate \
--dataset="ag_news" \
--pretrained_model_name_or_path="distilbert-base-uncased" \
--metric="accuracy" \
--querier="random" \
--stopper="null" \
--n_iterations=4 \
--log_level="warning" \
--n_start=64 \
--n_query=64 \
--output_dir="WILL_BE_IGNORED" \
--learning_rate="2e-5" \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=768 \
--num_train_epochs=25 \
--weight_decay=0.01 \
--evaluation_strategy="epoch" \
--save_strategy="epoch" \
--load_best_model_at_end \
--save_total_limit=1 \
--optim="adamw_torch" \
--group_by_length \
--fp16=True
