#!/bin/bash

# Encode query from the trec dl test set
# $1: the directory where query and corpus embeddings are stored
# $2: path to the model
# $3: year, 2019 / 2020


mkdir -p $1

python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $2/model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name json \
  --encode_in_path trec/trec_$3_query.jsonl \
  --encoded_save_path $1/query_emb.trec.$3.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --config_name bert-base-uncased
