#!/bin/bash

# Merge all distributed search intermediate results to get the final ranking result
# $1: encoding path
# $2: result save path
# $3: year, 2019 / 2020

python -m tevatron.faiss_retriever.reducer \
    --score_dir $1/intermediate_trec_$3 \
    --query $1/query_emb.trec.$3.pkl \
    --save_ranking_to $2/rank.txt
