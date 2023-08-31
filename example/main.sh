#!/bin/bash -l

#SBATCH --job-name="al-clf"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate MDALTH


export CUDA_VISIBLE_DEVICES=0


QUERIER="uncertainty"
STOPPER="null"
NSTART=0.10
NQUERY=0.05
SUBSET_TRAIN=16384
SUBSET_TEST=4096
OUTPUT="/tmp/IGNORED"
NTRAIN_EPOCHS=32
STRATEGY="epoch"
NWORKERS=16
SAVE_TOTAL_LIMIT=1
OPTIM="adamw_torch"
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
LOG_STRATEGY="no"
LOG_LEVEL="critical"


echo "----------------------------------------"
echo "Begining text classification experiments"
echo "----------------------------------------"

# torchrun --standalone --nnodes=1 --nproc-per-node=2 \
python \
./example/main.py \
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
--per_device_eval_batch_size=512 \
--gradient_accumulation_steps=4 \
--num_train_epochs=$NTRAIN_EPOCHS \
--evaluation_strategy=$STRATEGY \
--save_strategy=$STRATEGY \
--logging_strategy=$LOG_STRATEGY \
--save_total_limit=$SAVE_TOTAL_LIMIT \
--optim=$OPTIM \
--weight_decay=$WEIGHT_DECAY \
--warmup_ratio=$WARMUP_RATIO \
--dataloader_num_workers=$NWORKERS \
--log_level=$LOG_LEVEL \
--dataloader_pin_memory \
--load_best_model_at_end \
--auto_find_batch_size \
--fp16


echo "-----------------------------------------"
echo "Begining image classification experiments"
echo "-----------------------------------------"

# torchrun --standalone --nnodes=1 --nproc-per-node=2 \
python \
./example/main.py \
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
--per_device_eval_batch_size=128 \
--gradient_accumulation_steps=16 \
--num_train_epochs=$NTRAIN_EPOCHS \
--evaluation_strategy=$STRATEGY \
--save_strategy=$STRATEGY \
--logging_strategy=$LOG_STRATEGY \
--save_total_limit=$SAVE_TOTAL_LIMIT \
--optim=$OPTIM \
--weight_decay=$WEIGHT_DECAY \
--warmup_ratio=$WARMUP_RATIO \
--dataloader_num_workers=$NWORKERS \
--log_level=$LOG_LEVEL \
--dataloader_pin_memory \
--load_best_model_at_end \
--auto_find_batch_size \
--fp16


echo "-----------------------------------------"
echo "Begining audio classification experiments"
echo "-----------------------------------------"

# torchrun --standalone --nnodes=1 --nproc-per-node=2 \
python \
./example/main.py \
--task="audio" \
--learn \
--evaluate \
--dataset="speech_commands" \
--pretrained_model_name_or_path="facebook/wav2vec2-base" \
--querier=$QUERIER \
--stopper=$STOPPER \
--n_start=$NSTART \
--n_query=$NQUERY \
--subset_train=$SUBSET_TRAIN \
--subset_test=$SUBSET_TEST \
--output_dir=$OUTPUT \
--per_device_train_batch_size=128 \
--per_device_eval_batch_size=256 \
--gradient_accumulation_steps=8 \
--num_train_epochs=$NTRAIN_EPOCHS \
--evaluation_strategy=$STRATEGY \
--save_strategy=$STRATEGY \
--logging_strategy=$LOG_STRATEGY \
--save_total_limit=$SAVE_TOTAL_LIMIT \
--optim=$OPTIM \
--weight_decay=$WEIGHT_DECAY \
--warmup_ratio=$WARMUP_RATIO \
--dataloader_num_workers=$NWORKERS \
--log_level=$LOG_LEVEL \
--dataloader_pin_memory \
--load_best_model_at_end \
--auto_find_batch_size \
--fp16
