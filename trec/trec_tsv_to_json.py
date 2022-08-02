import csv
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_file", type=str, metavar='path', required=True)
    parser.add_argument("--json_file", type=str, metavar='path', required=True)

    args = parser.parse_args()
    reader = csv.reader(open(args.trec_file), delimiter='\t')

    with open(args.json_file, 'w') as f:
        for line in reader:
            data = {
                "query_id": line[0],
                "query": line[1],
            }
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
