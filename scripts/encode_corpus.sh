#!/bin/bash

# Encode one shard of the whole corpus, store results in $1
# args:
#   $1: encoding store directionary
#   $2: number of shard of the corpus to be encoded
#   $3: model path

export CUDA_VISIBLE_DEVICES=0

EMBEDDING_DIR=$1
NUM_SHARD=$2
MODEL_DIR=$3

MODEL_NAME=$4 # dce / mvr / mebert

mkdir -p $EMBEDDING_DIR


for ((i=0; i<$NUM_SHARD; i++)); do
  python -m mvdr.models.$MODEL_NAME.encode \
    --output_dir=temp \
    --model_name_or_path $MODEL_DIR/model_msmarco \
    --fp16 \
    --per_device_eval_batch_size 128 \
    --p_max_len 128 \
    --encode_in_path data/msmarco_corpus_with_query/corpus.jsonl \
    --encoded_save_path $EMBEDDING_DIR/corpus_emb.$i.pkl \
    --encode_num_shard $NUM_SHARD \
    --encode_shard_index $i \
    --num_query 10 \
    --config_name bert-base-uncased
done
