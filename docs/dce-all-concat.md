
To replicate this baseline, you need different training and corpus encoding scripts.
The query encoding and search code can be reused.


### Data preprocessing

Prepare the training data where each document is attached with T5 generated queries.

```bash
bash data_scripts/generated_d2q.sh

python data_scripts/add_query_to_train.py --doc2query_file doc2query.tsv --save_path data/msmarco_train_with_query
```

### Training


```bash
export MODEL_DIR=/path/to/save/model
export MODEL_NAME=de
bash scripts_all_concat/train.sh $MODEL_DIR $MODEL_NAME
```


### Encoding

The following code encode the corpus into vectors. The corpus is partitioned into 80 shards due to resource limit.

```bash
export ENCODE_DIR=/path/to/save/encoding
export NUM_SHARDS=80
# encode corpus
bash scripts_all_concat/encode_corpus.sh $ENCODE_DIR $NUM_SHARDS $MODEL_DIR $MODEL_NAME
```
