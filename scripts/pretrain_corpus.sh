#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 -m tevatron.driver.train_corpus \
  --output_dir $1/model_msmarco \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name jordane95/msmarco-passage-corpus-with-query \
  --fp16 \
  --per_device_train_batch_size 256 \
  --train_n_passages 1 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --overwrite_output_dir
