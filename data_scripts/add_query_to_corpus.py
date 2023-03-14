import os
import argparse
from tqdm import tqdm

from datasets import load_dataset


def load_doc2queries(path: str = None):
    """
    Args:
        path (str): path to the doc2query file
    Returns:
        doc2queries (List[List[str]]): all generated queries
    """
    doc2queries = []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            doc2queries.append(line.strip().split('\t'))
    return doc2queries


def main():
    parser = argparse.ArgumentParser(description="Adding generated queries to the corpus.")
    parser.add_argument("--dataset_name", type=str, default="Tevatron/msmarco-passage-corpus", help="Name of huggingface dataset.")
    parser.add_argument("--doc2query_file", type=str, help="Path to the tsv format doc2queries file.", required=True)
    parser.add_argument("--num_query", type=int, default=10, help="Number of queries to preserve.")
    parser.add_argument("--save_path", type=str, help="Path to store the new dataset with generated query.", required=True)

    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_dataset(args.dataset_name, split='train')

    print("Loading doc2queries from {}".format(args.doc2query_file))
    doc2queries = load_doc2queries(args.doc2query_file)

    print("Adding generated queries to the corpus...")
    corpus = corpus.add_column("queries", doc2queries)

    print("Saving new corpus to {}".format(args.save_path))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    corpus.to_json(os.path.join(args.save_path, "corpus.jsonl"))


if __name__ == "__main__":
    main()
