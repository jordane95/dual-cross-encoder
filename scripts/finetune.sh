#!/bin/bash

# $1: model save path
# $2: init model path

python -m torch.distributed.launch --nproc_per_node=8 -m tevatron.driver.train \
  --output_dir $1/model_msmarco \
  --model_name_or_path $2/model_msmarco \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 20 \
  --logging_steps 500 \
  --overwrite_output_dir
