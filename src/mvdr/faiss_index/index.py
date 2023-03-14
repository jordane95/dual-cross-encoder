
import time
import math
import faiss
import os
import math
import pickle
from itertools import chain
from tqdm import tqdm
import argparse
import glob
import numpy as np


from .faiss_index_gpu import FaissIndexGPU


class FaissIndex():
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)

        return quantizer, index

    def train(self, train_data):
        print(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        print(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        print(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)


def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_sample(embeddings, sample_fraction=None):
    if sample_fraction:
        # import pdb; pdb.set_trace()
        sample = embeddings[np.random.randint(0, high=embeddings.shape[0], size=(int(embeddings.shape[0] * sample_fraction),))]
    else:
        sample = embeddings.copy()
    print("#> Sample has shape", sample.shape)
    return sample


def prepare_faiss_index(embeddings, partitions: int, sample_fraction: float = None):
    training_sample = load_sample(embeddings, sample_fraction=sample_fraction)

    dim = training_sample.shape[-1]
    index = FaissIndex(dim, partitions)

    print("#> Training with the vectors...")

    index.train(training_sample)

    print("Done training!\n")

    return index

def get_faiss_index_name(args, offset=None, endpos=None):
    partitions_info = '' if args.partitions is None else f'.{args.partitions}'
    range_info = '' if offset is None else f'.{offset}-{endpos}'

    return f'ivfpq{partitions_info}{range_info}.faiss'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default='embeddings_time')
    parser.add_argument('--sample', type=float, default=0.3)
    parser.add_argument('--partitions', type=int, default=None)
    parser.add_argument('--index_name', type=str, default=None)

    return parser.parse_args()


def index_faiss(args):
    print("#> Starting..")

    index_files = glob.glob(args.index_path) # List[str], all paths to document representation
    p_reps_0, p_lookup_0 = pickle_load(index_files[0])

    num_embeddings = p_reps_0.shape[0] * len(index_files)

    if args.partitions is None:
        args.partitions = 1 << math.ceil(math.log2(8 * math.sqrt(num_embeddings)))
        print('\n\n')
        print("You did not specify --partitions!")
        print("Default computation chooses", args.partitions,
                    "partitions (for {} embeddings)".format(num_embeddings))
        print('\n\n')

    index = prepare_faiss_index(p_reps_0, args.partitions, args.sample)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))

    print("#> Indexing the vectors...")

    look_up = []
    for p_reps, p_lookup in shards:
        index.add(p_reps)
        look_up += p_lookup

    print("Done indexing!")
    faiss_index_name = f'ivfpq.{args.partitions}.faiss' if args.index_name is None else args.index_name
    output_path = os.path.join(os.path.dirname(args.index_path), faiss_index_name)

    assert not os.path.exists(output_path), output_path
    index.save(output_path)

    print(f"\n\nDone! All complete!")


if __name__ == "__main__":
    index_faiss(get_args())
