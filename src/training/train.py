# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def identity(x):
    return x


class SplitSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        split_length (int): Split long sequences in sequences of max length n
    """
    def __init__(self, dataset, indices, split_length):
        self.dat = dataset
        self.n = split_length
        if indices is None:
            indices = range(len(dataset))
        self.ind = [(idx, i) for idx in indices
                    for i in range(math.ceil(len(self.dat[idx]) / self.n))]

    def __getitem__(self, idx):
        idx, i = self.ind[idx]
        x = self.dat[idx]
        if i == 0:
            return x[:self.n]
        return x[(i - 1) * self.n:i * self.n]

    def __len__(self):
        return len(self.ind)


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """
    def __init__(self, data, sort_key=identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together examples with a similar
        size length to reduce the padding required for each batch while maintaining some noise
        through bucketing.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.

    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123)
        >>>
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key=identity,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)),
            False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in SubsetRandomSampler(
                    list(
                        BatchSampler(sorted_sampler, self.batch_size,
                                     self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


def prepare_batch(sequences):
    sequences = [torch.from_numpy(np.asarray(s)) for s in sequences]
    input_ids = torch.nn.utils.rnn.pad_sequence(sequences,
                                                batch_first=True,
                                                padding_value=0)
    mask = torch.zeros(input_ids.shape, dtype=torch.float)
    for i, seq in enumerate(sequences):
        mask[i, :len(seq)] = 1.
    return {
        'input_ids': input_ids,
        'labels': input_ids,
        'attention_mask': mask
    }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from s3"
        })


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(default=None,
                                    metadata={"help": "The input data."})

    block_size: int = field(
        default=-1,
        metadata={
            "help":
            "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )


class MyTrainer(Trainer):
    def get_train_dataloader(self):
        subsampler = RandomSampler(
            self.train_dataset
        ) if self.args.local_rank == -1 else DistributedSampler(
            self.train_dataset, shuffle=True)
        sampler = BucketBatchSampler(
            subsampler,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            sort_key=lambda idx: len(self.train_dataset[idx]),
            bucket_size_multiplier=100)
        loader = DataLoader(self.train_dataset,
                            batch_sampler=sampler,
                            collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers)
        return loader

    def get_eval_dataloader(self):
        subsampler = SequentialSampler(
            self.eval_dataset
        ) if self.args.local_rank == -1 else DistributedSampler(
            self.eval_dataset, shuffle=False)
        sampler = BucketBatchSampler(
            subsampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            sort_key=lambda idx: len(self.eval_dataset[idx]),
            bucket_size_multiplier=100)
        loader = DataLoader(self.eval_dataset,
                            batch_sampler=sampler,
                            collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers)
        return loader


class TokensDataset(Dataset):
    def __init__(self, data_path, data_index_path, mmap, subset_size):
        self.data_path = data_path
        self.data_index_path = data_index_path
        self.mmap = mmap
        self.subset_size = subset_size

        if not os.path.exists(data_index_path):
            print(f'WARNING: {data_index_path} does not exist')
        if not os.path.exists(data_path):
            print(f'WARNING: {data_path} does not exist')

        self.ind = np.load(data_index_path) if os.path.exists(
            data_index_path) else None
        self.dat = np.load(data_path, mmap_mode='r' if mmap else
                           None) if os.path.exists(data_index_path) else None

    def train(self, split_length):
        data_index_path = self.data_index_path.replace('index', 'index-train')

        if self.dat is None:
            data_path = Path(self.data_path)
            data_path = data_path.parent / data_path.name.replace(
                'data', 'data-train')
            print(f'training data from {data_path} [{data_index_path}]')
            dataset = TokensDataset(data_path, data_index_path, self.mmap,
                                    self.subset_size)
            return SplitSubset(dataset, None, split_length)

        ind = np.load(data_index_path)
        if self.subset_size < 1.0:
            perm = torch.randperm(len(ind)).tolist()
            ind = ind[perm[:int(self.subset_size * len(ind))]]
        return SplitSubset(self, ind, split_length)

    def valid(self, split_length):
        data_index_path = self.data_index_path.replace('index', 'index-valid')

        if self.dat is None:
            data_path = Path(self.data_path)
            data_path = data_path.parent / data_path.name.replace(
                'data', 'data-valid')
            print(f'validation data from {data_path} [{data_index_path}]')
            dataset = TokensDataset(data_path, data_index_path, self.mmap,
                                    self.subset_size)
            return SplitSubset(dataset, None, split_length)

        ind = np.load(data_index_path)
        return SplitSubset(self, ind, split_length)

    def __len__(self):
        return 0 if self.ind is None else len(self.ind)

    def __getitem__(self, idx):
        if idx == 0:
            x = self.dat[:self.ind[idx]]
        else:
            x = self.dat[self.ind[idx - 1]:self.ind[idx]]

        return np.array(x, dtype=np.int64)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('init')

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
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

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(config.vocab_size)

    if data_args.block_size <= 0:
        data_args.block_size = config.n_positions
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, config.n_positions)

    print('get data')
    # Get datasets
    dataset = TokensDataset(f'{data_args.data_dir}/data.npy',
                            f'{data_args.data_dir}/index.npy',
                            mmap=False,
                            subset_size=1.0)
    train_dataset = dataset.train(data_args.block_size)
    eval_dataset = dataset.valid(data_args.block_size)

    print('start')
    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=prepare_batch,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    if training_args.do_train:
        model_path = (model_args.model_name_or_path
                      if model_args.model_name_or_path is not None
                      and os.path.isdir(model_args.model_name_or_path) else
                      None)
        trainer.train(model_path=model_path)
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir,
                                        "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


if __name__ == "__main__":
    main()
