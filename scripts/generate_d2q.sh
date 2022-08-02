#!/bin/bash

wget https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/predicted_queries_topk_sampling.zip

unzip predicted_queries_topk_sampling.zip

for i in $(seq -f "%03g" 0 17); do
    echo "Processing chunk $i"
    paste predicted_queries_topk_sample00?.txt${i}-1004000 \
    > predicted_queries_topk.txt${i}-1004000
done

cat predicted_queries_topk.txt???-1004000 > doc2query.tsv
