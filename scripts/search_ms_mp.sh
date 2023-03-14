#!/bin/bash

# search query on one shard of the corpus
# :param $1 the directionary where query and corpus embeddings are stored
# :param $2 search shard number

EMBEDDING_DIR=$1
NSPLIT=$2

NPROCESS=8 # number of processes

SAVE_PATH=$EMBEDDING_DIR/intermediate
mkdir -p $SAVE_PATH

pids=()
for ((i=0; i<$NSPLIT; i++)); do
    python -m mvdr.faiss_retriever \
        --query_reps $EMBEDDING_DIR/query_emb.pkl \
        --passage_reps $EMBEDDING_DIR/corpus_emb.$SHARD.pkl \
        --depth 100 \
        --batch_size -1 \
        --save_ranking_to $SAVE_PATH/shard.$i.pkl \
        --num_query 10 &
    pids+=($!);
    if (( $i % $NPROCESS == 0 ))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done