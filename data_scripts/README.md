
# Data Preprocessing

This folder shows how to generated corpus files with generated queries.

1. Download generated queries for each document
    
    `bash generated_d2q.sh`

2. Generated corpus file with queries

    `python add_query_to_corpus.py --doc2query_file doc2query.tsv --save_path msmarco_corpus_with_query`

3. Generate training data with queries

    `python add_query_to_train.py --doc2query_file doc2query.tsv --save_path msmarco_trainset_with_query`
