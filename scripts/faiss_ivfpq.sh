# index compression techniques used in colbert

export CUDA_VISIBLE_DEVICES=1

embeddings=embeddings_time

python -m tevatron.faiss_index.index \
    --index_path 'embeddings_time/corpus_emb.*.pkl' \
    --partitions 32768 \
    --sample 0.3 \
    --index_name ivfpq.32768.faiss
