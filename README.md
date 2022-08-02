# Dual Cross Encoder

Learning Diverse Document Representations with Deep Query Interactions for Dense Retrieval

## Environment Setup

Make sure you have a `python>=3.7` env with `pytorch` installed.
Then run the following command to setup the environment.

```bash
pip install -e .
```

## Experiments

### Training

There are two ways to train the model. One uses the query generation as data augmentation and the other does not.

> Note: Our models are trained on 8 V100 GPUs with 32G memory. If you use differenty configurations, please change the parameters in the training scripts accordingly.

* w/o data augmentation
```bash
MODEL_DIR=/path/to/save/model
bash scripts/train.sh $MODEL_DIR
```

* w/ data augmentation
```bash
PRETRAINED_MODEL_DIR=/path/to/save/pretrained/model
MODEL_DIR=/path/to/save/model
bash scripts/pretrain_corpus.sh $PRETRAINED_MODEL_DIR
bash scripts/finetune.sh $MODEL_DIR $PRETRAINED_MODEL_DIR
```

### Encoding

The following code encode the corpus into vectors. The corpus is partitioned into 20 shards due to resource limit.

```bash
ENCODE_DIR=/path/to/save/encoding
# encode corpus
for i in $(seq 0 19)
do
bash scripts/encode_corpus_with_query_shard.sh $ENCODE_DIR $i $MODEL_DIR
done
```

### Retrieval

We evaluate the retrieval performance on the following two benchmarks.

* MS MARCO

```bash
RESULT_DIR=/path/to/save/result

# encode query
bash scripts/encode_dev_query.sh $ENCODE_DIR $MODEL_DIR

# shard search
for i in $(seq 0 19)
do
bash scripts/search_shard.sh $ENCODE_DIR $i
done

# reduce
bash scripts/reduce.sh $ENCODE_DIR $RESULT_DIR

# evaluation
bash scripts/evaluate.sh $RESULT_DIR
```

* TREC DL

```bash
YEAR=2019 # 2020
RESULT_DIR=/path/to/save/result

# encode query
bash scripts/encode_trec_query.sh $ENCODE_DIR $MODEL_DIR $YEAR

# shard search
for i in $(seq 0 19)
do
bash scripts/search_trec_shard.sh $ENCODE_DIR $i $YEAR
done

# reduce
bash scripts/reduce_trec.sh $ENCODE_DIR $RESULT_DIR $YEAR

# evaluation
bash scripts/evaluate_trec.sh $RESULT_DIR $YEAR
```

## Acknowledgement

The code is mainly based on the [Tevatron](https://github.com/texttron/tevatron) toolkit. We also used some code and data from [docTTTTTquery](https://github.com/castorini/docTTTTTquery), [beir](https://github.com/beir-cellar/beir) and [transformers](https://github.com/huggingface/transformers). Thanks for the great work!
