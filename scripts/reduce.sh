#!/bin/bash

# Merge all distributed search intermediate results to get the final ranking result
# $1: encoding path
# $2: result save path

python -m tevatron.faiss_retriever.reducer \
    --score_dir $1/intermediate \
    --query $1/query_emb.pkl \
    --save_ranking_to $2/rank.txt
