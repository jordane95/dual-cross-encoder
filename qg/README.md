# Query Generation

## Trainining
```bash
python run_doc2query.py \
    --model_name_or_path t5-base \
    --output_dir t5_d2q \
    --dataset_name Tevatron/msmarco-passage \
    --do_train \
    --per_device_train_batch_size 8 \
    --save_steps 10000
```

## Inference
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python query_gen.py \
    --dataset Tevatron/msmarco-passage-corpus \
    --output_dir qgen_corpus \
    --ques_per_passage 10 \
    --batch_size 32
```
