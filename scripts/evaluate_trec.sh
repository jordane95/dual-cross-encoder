#!/bin/bash

# $1: result save path
# $2: year, 2019 / 2020

RESULT_DIR=$1
TREC_DIR=tmp/trec
YEAR=$2

mkdir -p $TREC_DIR

wget -P $TREC_DIR https://trec.nist.gov/data/deep/${YEAR}qrels-pass.txt

python trec/trec_test_eval.py \
    --reference $TREC_DIR/${YEAR}qrels-pass.txt \
    --retrieval $RESULT_DIR/rank.txt
