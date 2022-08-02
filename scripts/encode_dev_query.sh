#!/bin/bash

# Encode query in the dev set of msmarco

mkdir -p $1

python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $2/model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path $1/query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --config_name bert-base-uncased
