#!/bin/bash

# Merge all distributed search intermediate results to get the final ranking result
# $1: encoding path
# $2: result save path
# $3: year, 2019 / 2020

EMBEDDING_DIR=$1
RESULT_DIR=$2
YEAR=$3

python -m mvdr.faiss_retriever.reducer \
    --score_dir $EMBEDDING_DIR/intermediate_trec_$YEAR \
    --query $EMBEDDING_DIR/query_emb.trec.$YEAR.pkl \
    --save_ranking_to $RESULT_DIR/rank.txt
