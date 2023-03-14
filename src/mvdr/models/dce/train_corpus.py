import logging
import os
import sys

from typing import Tuple, List

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import PreTrainedTokenizer, BatchEncoding


from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from .data import QPCollator
from .modeling import DenseModel
from .trainer import DenseTrainer as Trainer, GCTrainer
from .dataset import HFCorpusDataset

import datasets
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TrainCorpusDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: Trainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: str, is_query=False):
        """add [CLS] and [SEP], no padding, with truncation, donot return mask and token type ids
        """
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # BatchEncoding
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item] # Dict[str, Union[List[int], List[List[int]]]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed) # int

        # Random fetch a query for each passage
        qrys = group['queries'] # List[str]
        qry = qrys[(_hashed_seed + epoch) % len(qrys)] # str

        encoded_query = self.create_one_example(qry, is_query=True) # BatchEncoding

        pos_psg = group['title'] + " " + group['text']
        encoded_passages = self.create_one_example(pos_psg)

        return encoded_query, encoded_passages


def main():
    # Initialize a huggingface argument parser with 3 self-defined dataclass types
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # load args from json file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # load args from command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    # Load model
    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Load a huggingface Dataset object
    train_corpus_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    # data is still text in this dataset

    # convert it to a torch Dataset object
    train_dataset = TrainCorpusDataset(data_args, train_corpus_dataset.dataset, tokenizer)
    
    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
