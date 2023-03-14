#!/bin/bash

# $1: result save path

RESULT_DIR=$1

python -m mvdr.utils.format.convert_result_to_marco \
    --input $RESULT_DIR/rank.txt \
    --output $RESULT_DIR/rank.txt.marco

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset $RESULT_DIR/rank.txt.marco
