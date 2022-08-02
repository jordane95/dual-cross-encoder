import logging
import argparse
from typing import List, Dict, Tuple

import pytrec_eval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def load_reference(path: str) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(path, 'r') as f:
        for line in f:
            qid, Q0, docid, score = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(score)
    return qrels


def load_retrieval_results(path: str) -> Dict[str, Dict[str, float]]:
    results = {}
    with open(path, 'r') as f:
        for line in f:
            qid, pid, score = line.strip().split('\t')
            if qid not in results:
                results[qid] = {}
            results[qid][pid] = float(score)
    return results


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Args:
        qrels : ground truth
    """
    ndcg = {} # Dict[str, float]
    _map = {}
    recall = {}
    precision = {}
    mrr = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
        mrr[f"MRR@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values]) # "map_cut.1,5,10" for example
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    mrr_string = "recip_rank"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, mrr_string})
    scores = evaluator.evaluate(results) # Dict[str, Dict[str, float]]: maps qid to its score which is a dict maps score name to score
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
            mrr[f"MRR@{k}"] += scores[query_id]["recip_rank"]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        mrr[f"MRR@{k}"] = round(mrr[f"MRR@{k}"]/len(scores), 5)
    
    for eval in [ndcg, _map, recall, precision, mrr]:
        # eval: Dict[str, float]
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, metavar='path', required=True)
    parser.add_argument("--reference", type=str, metavar='path', required=True)
    args = parser.parse_args()

    qrels = load_reference(args.reference)
    results = load_retrieval_results(args.retrieval)

    ndcg, _map, recall, precision = evaluate(qrels, results, k_values=[1,3,5,10,100,1000])


if __name__ == "__main__":
    main()
