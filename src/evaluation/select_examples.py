from argparse import ArgumentParser
from pathlib import Path
import os

from tqdm import tqdm
import random


def select_n_examples(f, n, min_length=100):
    examples = []

    for line in tqdm(f):
        if len(line) >= min_length:
            examples.append(line)

    print(f' > read {len(examples)} lines')
    examples = random.sample(examples, n)

    return examples


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('--src', default='small', choices=['full', 'small'])
    parser.add_argument('--file', default='full')
    parser.add_argument('-n', default=100, type=int)
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    fname = args.file + '.txt'
    src_path = Path('data') / args.lang / 'evaluation' / 'plaintext' / args.src / fname
    if not src_path.exists():
        print(f' > source path {src_path} does not exist')
        exit(1)

    tgt_path = Path('data') / args.lang / 'evaluation' / 'examples' / args.src / args.file / 'gold.txt'
    os.makedirs(tgt_path.parent, exist_ok=True)

    if not args.force and tgt_path.exists():
        print(f'{tgt_path} already exists. use -f to override')
        exit(1)

    print(f' > selecting {args.n} examples from {src_path}')
    with open(src_path) as f:
        examples = select_n_examples(f, args.n)
    assert len(examples) == args.n

    with open(tgt_path, 'w') as f:
        f.writelines(examples)

    print(f'\nsaved {args.n} examples to {tgt_path}')




if __name__ == '__main__':
    main()
