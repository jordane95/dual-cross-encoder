export CUDA_VISIBLE_DEVICES=1

embeddings=embeddings_time

mkdir -p $embeddings/intermediate

i=0
python -m mvdr.faiss_index.search \
    --query_reps $embeddings/query_emb.pkl \
    --passage_reps $embeddings/corpus_emb.$i.pkl \
    --depth 1000 \
    --batch_size -1 \
    --save_ranking_to $embeddings/intermediate/shard.$i.pkl \
    --num_query 10 \
    --index_path $embeddings/ivfpq32768.faiss
