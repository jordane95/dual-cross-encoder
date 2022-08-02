#!/bin/bash

# search query on one shard of the corpus
# :param $1 the directionary where query and corpus embeddings are stored
# :param $2 search shard number

mkdir -p $1/intermediate

python -m tevatron.faiss_retriever \
    --query_reps $1/query_emb.pkl \
    --passage_reps $1/corpus_emb.$2.pkl \
    --depth 100 \
    --batch_size -1 \
    --save_ranking_to $1/intermediate/shard.$2.pkl \
    --num_query 10
