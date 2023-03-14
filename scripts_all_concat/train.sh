
MODLE_DIR=$1
MODEL_NAME=$2

python -m torch.distributed.launch --nproc_per_node=8 -m mvdr.models.de.train \
  --output_dir $MODEL_DIR/model_msmarco \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --train_dir data/msmarco_train_with_query \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 156 \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --num_query 10 \
  --all_concat

# --all_concat False will be the dual encoder baseline
 