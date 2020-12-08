from argparse import ArgumentParser
from pathlib import Path
import pickle
import os

from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.processors import RobertaProcessing
from transformers import AutoTokenizer


def init_tokenizer(lang, n, m):
    if n is None and m is None:
        print('size nor model are specified, but one of them is required')
        exit(1)

    if m is not None:
        tokenizer = AutoTokenizer.from_pretrained(m, use_fast=True)
        return tokenizer

    tokenizer = Tokenizer.from_file(
        str(
            Path('data') / lang / 'preparation' / 'vocabularies' /
            f'{lang}-{str(n).zfill(3)}k.tokenizer.json'))
    tokenizer.post_processor = RobertaProcessing(
        ('</s>', tokenizer.token_to_id('</s>')),
        ('<s>', tokenizer.token_to_id('<s>')),
        trim_offsets=True)
    return tokenizer


def tokenize_doc(tokenizer: Tokenizer, doc):
    enc = tokenizer.encode(doc)
    if type(enc) == list:
        return enc
    return enc.ids


def tokenize_file(tokenizer, src_path, eot=None):
    examples = []
    doc = ''
    with open(src_path) as f:
        for line in tqdm(f):
            if eot is None and line == '\n':
                examples.append(tokenize_doc(tokenizer, doc))
                doc = ''
                continue
            elif eot is not None and line == eot + '\n':
                examples.append(tokenize_doc(tokenizer, doc.strip()))
                doc = ''
                continue

            doc += line

    if doc != '':
        examples.append(tokenize_doc(tokenizer, doc))
    return examples


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('--size',
                        type=int,
                        default=None,
                        help='vocab size (in thousands)')
    parser.add_argument('--model',
                        default=None,
                        help='HuggingFace model identifier')
    parser.add_argument('--eot', default=None)
    args = parser.parse_args()

    prep_dir = Path('data') / args.lang / 'preparation' / 'prepared'

    dst_path = prep_dir / ('data.pkl' if args.size is None else
                           f'data-{str(args.size).zfill(3)}k.pkl')
    if not dst_path.parent.exists():
        os.makedirs(dst_path.parent)

    print(f' > preparing {dst_path}')
    tokenizer = init_tokenizer(args.lang, args.size, args.model)

    examples = []

    src_paths = sorted((Path('data') / args.lang / 'preparation' /
                        'plaintext').glob('**/*.txt'))
    for src_path in src_paths:
        print('🔥', src_path)
        new_examples = tokenize_file(tokenizer, src_path, eot=args.eot)

        if src_path.name in ['train.txt', 'valid.txt', 'test.txt']:
            subset = src_path.name.split('.')[0]
            out_path = dst_path.parent / dst_path.name.replace(
                'data', f'data-{subset}')
            print(f' > exporting {len(new_examples):,} examples to {out_path}')
            with open(out_path, 'wb') as f:
                pickle.dump(new_examples, f)

        examples.extend(new_examples)
        print(f' ::: {len(examples):,} examples loaded')

    print(f'{len(examples):,} examples')

    print(f' > exporting {dst_path}')
    with open(dst_path, 'wb') as f:
        pickle.dump(examples, f)


if __name__ == '__main__':
    main()
