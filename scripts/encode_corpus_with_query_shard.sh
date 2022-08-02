#!/bin/bash

# Encode one shard of the whole corpus, store results in $1
# args:
#   $1: encoding store directionary
#   $2: number of shard of the corpus to be encoded
#   $3: model path

mkdir -p $1

python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $3/model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --p_max_len 128 \
  --dataset_name jordane95/msmarco-passage-corpus-with-query \
  --encoded_save_path $1/corpus_emb.$2.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $2 \
  --num_query 10 \
  --config_name bert-base-uncased
