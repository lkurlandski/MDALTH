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
conda activate MDALTH

export CUDA_VISIBLE_DEVICES=0

export QUERIER="uncertainty"
export STOPPER="null"
export NSTART=0.10
export NQUERY=0.05
export NTRAIN_EPOCHS=32
export STRATEGY="epoch"
export NWORKERS=16
export OPTIM="adamw_torch"
export SAVE_TOTAL_LIMIT=1
export OUTPUT="WILL_BE_IGNORED"
export SUBSET_TRAIN=16384
export SUBSET_TEST=4096


echo "Begining text classification experiments"
echo "----------------------------------------"

# # torchrun --standalone --nnodes=1 --nproc-per-node=1
python ./example/main.py \
--task="text" \
--learn \
--evaluate \
--dataset="ag_news" \
--pretrained_model_name_or_path="distilbert-base-uncased" \
--querier=$QUERIER \
--stopper=$STOPPER \
--n_start=$NSTART \
--n_query=$NQUERY \
--subset_train=$SUBSET_TRAIN \
--subset_test=$SUBSET_TEST \
--output_dir=$OUTPUT \
--per_device_train_batch_size=256 \
--per_device_eval_batch_size=1024 \
--gradient_accumulation_steps=4 \
--num_train_epochs=$NTRAIN_EPOCHS \
--evaluation_strategy=$STRATEGY \
--save_strategy=$STRATEGY \
--logging_strategy=$STRATEGY \
--save_total_limit=$SAVE_TOTAL_LIMIT \
--optim=$OPTIM \
--dataloader_num_workers=$NWORKERS \
--dataloader_pin_memory \
--load_best_model_at_end \
--auto_find_batch_size \
--fp16=True


echo "Begining image classification experiments"
echo "-----------------------------------------"

# # torchrun --standalone --nnodes=1 --nproc-per-node=1 
python ./example/main.py \
--task="image" \
--learn \
--evaluate \
--dataset="food101" \
--pretrained_model_name_or_path="google/vit-base-patch16-224-in21k" \
--querier=$QUERIER \
--stopper=$STOPPER \
--n_start=$NSTART \
--n_query=$NQUERY \
--subset_train=$SUBSET_TRAIN \
--subset_test=$SUBSET_TEST \
--output_dir=$OUTPUT \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=256 \
--gradient_accumulation_steps=16 \
--num_train_epochs=$NTRAIN_EPOCHS \
--evaluation_strategy=$STRATEGY \
--save_strategy=$STRATEGY \
--logging_strategy=$STRATEGY \
--save_total_limit=$SAVE_TOTAL_LIMIT \
--optim=$OPTIM \
--dataloader_num_workers=$NWORKERS \
--dataloader_pin_memory \
--load_best_model_at_end \
--auto_find_batch_size \
--fp16=True

echo "Begining audio classification experiments"
echo "-----------------------------------------"

# # torchrun --standalone --nnodes=1 --nproc-per-node=1
python ./example/main.py \
--task="audio" \
--learn \
--evaluate \
--dataset="PolyAI/minds14" \
--pretrained_model_name_or_path="facebook/wav2vec2-base" \
--querier=$QUERIER \
--stopper=$STOPPER \
--n_start=$NSTART \
--n_query=$NQUERY \
--subset_train=$SUBSET_TRAIN \
--subset_test=$SUBSET_TEST \
--output_dir=$OUTPUT \
--per_device_train_batch_size=128 \
--per_device_eval_batch_size=512 \
--gradient_accumulation_steps=8 \
--num_train_epochs=$NTRAIN_EPOCHS \
--evaluation_strategy=$STRATEGY \
--save_strategy=$STRATEGY \
--logging_strategy=$STRATEGY \
--save_total_limit=$SAVE_TOTAL_LIMIT \
--optim=$OPTIM \
--dataloader_num_workers=$NWORKERS \
--dataloader_pin_memory \
--load_best_model_at_end \
--auto_find_batch_size \
--fp16=True
