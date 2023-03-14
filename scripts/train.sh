#!/bin/bash

SAVE_DIR=$1
MODEL_NAME=$2

python -m torch.distributed.launch --nproc_per_node=8 -m mvdr.models.$MODEL_NAME.train \
  --output_dir $SAVE_DIR/model_msmarco \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --overwrite_output_dir
