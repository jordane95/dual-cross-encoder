#!/bin/bash

# Encode query from the trec dl test set
# $1: the directory where query and corpus embeddings are stored
# $2: path to the model
# $3: year, 2019 / 2020
export CUDA_VISIBLE_DEVICES=0

EMBEDDING_DIR=$1
MODEL_DIR=$2
YEAR=$3

MODEL_NAME=$4 # dce / mvr / mebert

mkdir -p $EMBEDDING_DIR

python -m mvdr.models.$MODEL_NAME.encode \
  --output_dir=temp \
  --model_name_or_path $MODEL_DIR/model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name json \
  --encode_in_path trec/trec_{$YEAR}_query.jsonl \
  --encoded_save_path $EMBEDDING_DIR/query_emb.trec.$YEAR.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --config_name bert-base-uncased
