import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm

from .retriever import BaseFaissIPRetriever

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    """
    Args:
        q_reps (numpy.ndarray): shape [num_queries, query_emb_dim]
        p_lookup (List[str]): maps faiss index to pid
        args:
    Returns:
        filtered_scores (numpy.ndarray, dtype=float32): shape [num_samples, args.depth], query index -> retrieved passages scores
        filtered_indices (List[List[str]]): shape [num_samples, args.depth], query index -> retrieved pids
    """
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth * args.num_query, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth * args.num_query)
    # all_indices contains indice from Indexer, not the passage index, although they may differ,
    # their corresponding passge indice may be identique

    # filtered_indices contains distinct passage indices

    # eliminate the duplicate docid for each query and garde only the top 100
    filtered_scores = []
    filtered_indices = []

    for duplicate_scores, duplicate_indices in zip(all_scores, all_indices):
        psg_indices_cache = set()
        distinct_indices = []
        distinct_scores = []
        for score, indice in zip(duplicate_scores, duplicate_indices):
            psg_indice = p_lookup[indice] # str
            if len(distinct_indices) == args.depth:
                break
            elif psg_indice in psg_indices_cache:
                continue
            else:
                psg_indices_cache.add(psg_indice)
                distinct_indices.append(psg_indice)
                distinct_scores.append(score)
        filtered_scores.append(distinct_scores)
        filtered_indices.append(distinct_indices)
    
    filtered_scores = np.array(filtered_scores)
    filtered_indices = np.array(filtered_indices)
    return filtered_scores, filtered_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--num_query', type=int, default=10)

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps) # List[str], all paths to document representation
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = BaseFaissIPRetriever(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    # look_up: List[int], maps from idx to pid
    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()
