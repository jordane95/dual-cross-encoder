#!/bin/bash

# search query on one shard embedding of the corpus
# :param $1 the directionary where query and corpus embeddings are stored
# :param $2 shard number
# :param $3 year, 2019 / 2020

SAVE_PATH=$1/intermediate_trec_$3
mkdir -p $SAVE_PATH

python -m tevatron.faiss_retriever \
    --query_reps $1/query_emb.trec.$3.pkl \
    --passage_reps $1/corpus_emb.$2.pkl \
    --depth 1000 \
    --batch_size -1 \
    --save_ranking_to $SAVE_PATH/shard.$2.pkl \
    --num_query 10
