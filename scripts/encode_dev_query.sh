#!/bin/bash

# Encode query in the dev set of msmarco

export CUDA_VISIBLE_DEVICES=0

EMBEDDING_DIR=$1
MODEL_DIR=$2

MODEL_NAME=$3 # dce / mvr / mebert

mkdir -p $EMBEDDING_DIR

python -m mvdr.models.$MODEL_NAME.encode \
  --output_dir=temp \
  --model_name_or_path $MODEL_DIR/model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path $EMBEDDING_DIR/query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --config_name bert-base-uncased
