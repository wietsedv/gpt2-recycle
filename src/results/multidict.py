from argparse import ArgumentParser
from pathlib import Path
import os


def load_reversed_vocab(lang, model):
    d = {}
    with open(Path('data') / lang / 'dictionaries' /
              f'{model}.words.tsv') as f:
        for line in f:
            tgt, src = line.split('\t')[:2]
            if src not in d:
                d[src] = set()
            d[src].add(tgt)
    return d


def main():
    parser = ArgumentParser()
    parser.add_argument('lang1')
    parser.add_argument('lang2')
    parser.add_argument('model')
    args = parser.parse_args()

    print(f'extracting paired dictionary for {args.lang1} and {args.lang2}\n')

    d1 = load_reversed_vocab(args.lang1, args.model)
    print(f'dictionary {args.lang1}: {len(d1):,} tokens')

    d2 = load_reversed_vocab(args.lang2, args.model)
    print(f'dictionary {args.lang2}: {len(d2):,} tokens')

    eng_toks = list(d1.keys() & d2.keys())
    print(f'intersection:   {len(eng_toks):,} tokens')

    out_path = Path(
        'data'
    ) / 'eng' / 'results' / 'data' / 'dictionaries' / f'{args.lang1}-{args.lang2}.tsv'
    os.makedirs(out_path.parent, exist_ok=True)

    with open(out_path, 'w') as f:
        for eng in eng_toks:
            l1 = sorted(d1[eng])
            l2 = sorted(d2[eng])
            f.write(f'{eng}\t{"/".join(l1)}\t{"/".join(l2)}\n')
    print(f'\nsaved to {out_path}')


if __name__ == '__main__':
    main()
