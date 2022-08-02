#!/bin/bash

# $1: result save path

python -m tevatron.utils.format.convert_result_to_marco \
    --input $1/rank.txt \
    --output $1/rank.txt.marco

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset $1/rank.txt.marco
