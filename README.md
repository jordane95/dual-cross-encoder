# Dual Cross Encoder

Code for paper: Learning Diverse Document Representations with Deep Query Interactions for Dense Retrieval

## Environment Setup

Make sure you have a `python>=3.7` env with `pytorch` installed.
Then run the following command to setup the environment.

```bash
pip install -e .
```

## Experiments

The following instructions could help you replicate the results of three multi-vector dense retrieval models: ME-BERT, MVR and DCE.

To replicate DCE where all queries are concatenated with each document rather than individually, see [doc](docs/dce-all-concat.md) for more details.

### Data preprocessing

Attach T5 generated queries to the corpus.

```bash
bash data_scripts/generated_d2q.sh

python data_scripts/add_query_to_corpus.py --doc2query_file doc2query.tsv --save_path data/msmarco_corpus_with_query
```

### Training

> Note: Our models are trained on 8 V100 GPUs with 32GB memory. If you use different configurations, please change the parameters in the training scripts accordingly.

```bash
export MODEL_NAME=dce # mebert / mvr
export MODEL_DIR=dce_retriever
bash scripts/train.sh $MODEL_DIR $MODEL_NAME
```


### Encoding

The following code encode the corpus into vectors. The corpus is partitioned into 80 shards due to memory limit.

```bash
export ENCODE_DIR=/path/to/save/encoding
export NUM_SHARDS=80
# encode corpus
bash scripts/encode_corpus.sh $ENCODE_DIR $NUM_SHARDS $MODEL_DIR $MODEL_NAME
```

### Retrieval

We evaluate the retrieval performance on the following two benchmarks.

* MS MARCO

```bash
export RESULT_DIR=/path/to/save/result

# encode query
bash scripts/encode_dev_query.sh $ENCODE_DIR $MODEL_DIR $MODEL_NAME

# search
bash scripts/search_ms_mp.sh $ENCODE_DIR $NUM_SHARDS

# reduce
bash scripts/reduce.sh $ENCODE_DIR $RESULT_DIR

# evaluation
bash scripts/evaluate.sh $RESULT_DIR
```

* TREC DL

```bash
export YEAR=2019 # 2020
export RESULT_DIR=/path/to/save/result

# encode query
bash scripts/encode_trec_query.sh $ENCODE_DIR $MODEL_DIR $YEAR $MODEL_NAME

# search
bash scripts/search_trec_mp.sh $ENCODE_DIR $i $YEAR

# reduce
bash scripts/reduce_trec.sh $ENCODE_DIR $RESULT_DIR $YEAR

# evaluation
bash scripts/evaluate_trec.sh $RESULT_DIR $YEAR
```

## Acknowledgement

The code is mainly based on the [Tevatron](https://github.com/texttron/tevatron) toolkit. We also used some code and data from [docTTTTTquery](https://github.com/castorini/docTTTTTquery), [beir](https://github.com/beir-cellar/beir) and [transformers](https://github.com/huggingface/transformers).
