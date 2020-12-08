#! /usr/bin/env python3

from argparse import ArgumentParser
import json
import os
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from tqdm.std import tqdm
from transformers import GPT2LMHeadModel, GPT2Config
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from tokenizers import Tokenizer

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler


def identity(x):
    return x


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


def prepare_batch(sequences):
    sequences = [torch.from_numpy(np.asarray(s)) for s in sequences]
    input_ids = torch.nn.utils.rnn.pad_sequence(sequences,
                                                batch_first=True,
                                                padding_value=0)
    mask = torch.zeros(input_ids.shape, dtype=torch.float)
    for i, seq in enumerate(sequences):
        mask[i, :len(seq)] = 1.
    return {'input_ids': input_ids, 'attention_mask': mask}


class EmbeddingTunerModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.d = None
        self.tokenizer = None

        # hotfixes
        if 'unfreeze' not in hparams:
            self.hparams.unfreeze = False
        if 'lang' not in hparams:
            self.hparams.lang = 'nld'

        autofix_paths(self.hparams)

        # GPT with LM head and correct embedding size
        with open(Path('data') / self.hparams.lang / 'config.json') as f:
            cfg = json.load(f)

        if self.hparams.unfreeze:
            self.n_unfreeze = 0
            if self.hparams.resume_from_checkpoint is not None:
                print('Resuming from checkpoint: unfreezing all layers')
                self.n_unfreeze = None

        config = GPT2Config.from_pretrained(self.hparams.pretrained_path,
                                            **cfg)
        if self.hparams.unfreeze and self.n_unfreeze is not None:
            config.torchscript = True
        self.m = GPT2LMHeadModel.from_pretrained(self.hparams.pretrained_path,
                                                 config=config)

        # Resize vocab
        self.m.resize_token_embeddings(self.hparams.vocab_size)

    @property
    def batch_size(self):
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.hparams.batch_size = batch_size

    @property
    def lr(self):
        return self.hparams.lr

    @lr.setter
    def lr(self, lr):
        self.hparams.lr = lr

    def setup(self, stage):
        self.d = TokensDataset(self.hparams.data_path,
                               self.hparams.data_index_path, self.hparams.mmap,
                               self.hparams.subset_size)

        if self.hparams.tokenizer_path is not None:
            os.environ['TOKENIZERS_PARALLELISM'] = str(True)
            self.tokenizer = Tokenizer.from_file(self.hparams.tokenizer_path)

        self.m.train()

        # Freeze layers if embeds only
        if self.hparams.wte_only:
            for p in self.m.parameters():
                p.requires_grad = False
            if stage == 'fit':
                self.m.transformer.wte.weight.requires_grad = True
        elif self.hparams.unfreeze and self.n_unfreeze is not None:
            for p in self.m.parameters():
                p.requires_grad = False

            for i in range(max(self.n_unfreeze, 1)):
                for p in self.m.transformer.h[-i].parameters():
                    p.requires_grad = True
        else:
            for p in self.m.parameters():
                p.requires_grad = True

        self.m.resize_token_embeddings(self.hparams.vocab_size)

    def forward(self, **kwargs):
        return self.m(**kwargs)

    def unfreeze_step(self):
        if self.n_unfreeze is None:
            return

        self.n_unfreeze += 1
        tqdm.write(f'unfreezing {self.n_unfreeze} layers')

        params = []
        if self.n_unfreeze > len(self.m.transformer.h):
            self.n_unfreeze = None
            self.unfreeze()
            params = list(self.m.parameters())
            self.m.config.torchscript = False
            self.m.tie_weights()
        else:
            for h in self.m.transformer.h[-self.n_unfreeze:]:
                for p in h.parameters():
                    p.requires_grad = True
                    params.append(p)
        self.trainer.optimizers[0].param_groups[0]['params'] = params

    # def on_train_start(self):
    #     if self.hparams.unfreeze:
    #         self.unfreeze_step()

    def training_step(self, batch, batch_idx):
        output = self(**batch, labels=batch['input_ids'], use_cache=False)

        loss = output[0]
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(**batch, labels=batch['input_ids'], use_cache=False)

        loss = output[0]
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        if self.tokenizer is not None:
            outputs = self.m.generate(max_length=100)
            txt = self.tokenizer.decode(outputs[0].tolist(),
                                        skip_special_tokens=False)
            self.logger.experiment.add_text('example', txt)

        if self.hparams.unfreeze:
            self.unfreeze_step()

        return {
            'val_loss': avg_val_loss,
            'log': {
                'avg_val_loss': avg_val_loss
            }
        }

    def configure_optimizers(self):
        # if self.hparams.wte_only:
        #     params = [self.m.transformer.wte.weight]
        # elif self.hparams.unfreeze:
        #     raise NotImplementedError('unfreezing optimizer is not yet implemented')
        # else:
        #     params = self.m.parameters()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            self.m.parameters()),
                                     lr=self.lr,
                                     amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        data = self.d.train(self.hparams.max_seq_length)
        print(f'training examples: {len(data):,}')

        subsampler = DistributedSampler(data, shuffle=True) \
            if self.hparams.distributed_backend is not None else RandomSampler(data)
        sampler = BucketBatchSampler(subsampler,
                                     batch_size=self.batch_size,
                                     drop_last=False,
                                     sort_key=lambda idx: len(data[idx]),
                                     bucket_size_multiplier=100)

        loader = DataLoader(data,
                            batch_sampler=sampler,
                            collate_fn=prepare_batch,
                            num_workers=self.hparams.num_workers)
        return loader

    def val_dataloader(self):
        data = self.d.valid(self.hparams.max_seq_length)
        print(f'validation examples: {len(data):,}')
        subsampler = DistributedSampler(data, shuffle=True) \
            if self.hparams.distributed_backend is not None else RandomSampler(data)
        sampler = BucketBatchSampler(subsampler,
                                     batch_size=self.batch_size,
                                     drop_last=False,
                                     sort_key=lambda idx: len(data[idx]),
                                     bucket_size_multiplier=100)

        loader = DataLoader(data,
                            batch_sampler=sampler,
                            collate_fn=prepare_batch,
                            num_workers=self.hparams.num_workers)
        return loader

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser],
                                add_help=False,
                                fromfile_prefix_chars='@')

        parser.add_argument('--num_workers', type=int, default=4)

        # Data
        parser.add_argument('--data_path', default=None)
        parser.add_argument('--data_index_path', default=None)
        parser.add_argument('--mmap', action='store_true')
        parser.add_argument('--max_seq_length', type=int, default=1024)

        # Model
        parser.add_argument(
            '--pretrained_path',
            default='gpt2')  # gpt2 gpt2-medium gpt2-large gpt2-xl
        parser.add_argument('--vocab_size', type=int, default=40_000)
        parser.add_argument('--tokenizer_path', default=None)
        parser.add_argument('--wte_only', action='store_true')
        parser.add_argument('--unfreeze', action='store_true')
        parser.add_argument('--reset_state', action='store_true')
        parser.add_argument('--subset_size', default=1.0, type=float)

        # Training
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=3)

        return parser


def search_hparams(model, trainer: pl.Trainer, args):
    import matplotlib.pyplot as plt

    # LR
    lr_finder = trainer.lr_find(model, min_lr=1e-10, num_training=2000)
    new_lr = lr_finder.suggestion()
    print(f'LR suggestion: {new_lr}')
    lr_finder.plot(suggest=True)

    path = Path('data') / args.lang / 'tmp' / 'lr.png'
    os.makedirs(path.parent, exist_ok=True)
    plt.savefig(str(path))


def get_trainer_kwargs(args):
    version = args.version
    if args.name is not None:
        version = f'{version}-{args.name}'
    if args.resume_from_checkpoint is not None and '/' not in args.resume_from_checkpoint:
        if args.resume_from_checkpoint.endswith(args.name):
            version = f'{args.version}-{args.resume_from_checkpoint}'
        else:
            version = f'{version}-{args.resume_from_checkpoint}'
    args.version = version

    os.makedirs(Path('data') / args.lang / 'runs' / args.version,
                exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=str(
        Path('data') / args.lang),
                                          name='runs',
                                          version=args.version)
    lr_logger = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint(filepath=None,
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=True,
                                          save_top_k=5,
                                          save_last=True,
                                          period=0)

    return {
        'logger': logger,
        'default_root_dir': 'data',
        'callbacks': [lr_logger],
        'checkpoint_callback': checkpoint_callback,
        'replace_sampler_ddp': False
    }


def autofix_paths(args):
    prep_path = Path('data') / args.lang / 'preparation'

    if args.pretrained_path is not None and args.pretrained_path.startswith(
            'data/models/'):
        args.pretrained_path = args.pretrained_path.replace(
            'data/models/', 'data/nld/models/')
    if args.data_path is not None and args.data_path.startswith(
            'data/preparation/'):
        args.data_path = args.data_path.replace('data/preparation/',
                                                'data/nld/preparation/')
    if args.data_index_path is not None and args.data_index_path.startswith(
            'data/preparation/'):
        args.data_index_path = args.data_index_path.replace(
            'data/preparation/', 'data/nld/preparation/')

    if args.data_path is None or not os.path.exists(args.data_path):
        args.data_path = str(prep_path / 'final' / 'data.npy')
    if args.data_index_path is None or not os.path.exists(
            args.data_index_path):
        args.data_index_path = str(prep_path / 'final' / 'index.npy')
    if args.tokenizer_path is not None:
        if not os.path.exists(args.tokenizer_path):
            args.tokenizer_path = str(prep_path / 'vocabularies' /
                                      'tokenizer.json')
        if not os.path.exists(args.tokenizer_path):
            args.tokenizer_path = None


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = EmbeddingTunerModel.add_argparse_args(parser)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--seed', type=int, default=7649832)
    parser.add_argument('--version',
                        default=os.environ.get('SLURM_JOB_ID', '0'))
    parser.add_argument('--name', default=None)
    parser.add_argument('--lang', required=True)

    args = parser.parse_args()
    trainer_kwargs = get_trainer_kwargs(args)

    print(f'starting: {args.version}')
    pl.seed_everything(args.seed)

    if 'unfreeze' not in args:
        args.unfreeze = False

    assert not (args.wte_only and args.unfreeze)

    if args.reset_state:
        args.unfreeze = False

    if args.resume_from_checkpoint is None:
        model = EmbeddingTunerModel(args)
    else:
        if '/' not in args.resume_from_checkpoint:
            args.resume_from_checkpoint = str(
                Path('data') / args.lang / 'runs' /
                args.resume_from_checkpoint / 'checkpoints' / 'last.ckpt')
        print(f'resuming from {args.resume_from_checkpoint}',
              os.path.exists(args.resume_from_checkpoint))
        model = EmbeddingTunerModel.load_from_checkpoint(
            args.resume_from_checkpoint, map_location='cpu', **vars(args))

    if args.reset_state:
        args.resume_from_checkpoint = None

    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)

    if args.verbose:
        print(model.hparams)
        print(model)
        print(model.m.config)

    if args.search:
        return search_hparams(model, trainer, args)

    trainer.fit(model)


if __name__ == '__main__':
    main()
