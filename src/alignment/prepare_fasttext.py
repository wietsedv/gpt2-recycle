from argparse import ArgumentParser
from pathlib import Path
import pickle

from tqdm import tqdm
import numpy as np
import editdistance

from .utils import load_embeddings, load_vocab


def load_fasttext_vectormap(path):
    vectors = {}
    # TODO pair with BPE vocab

    with open(path) as f:
        n, d = [int(v) for v in next(f).split(' ')]
        for line in tqdm(f, total=n):
            token, vec = line.rstrip().split(' ', maxsplit=1)
            vec = np.array([float(v) for v in vec.split(' ')])
            assert len(vec) == d
            vectors[token] = vec

    return vectors


def minimum_editdistance(token, candidates):
    best, min_distance, max_overlap = '</s>', 1000, 1000
    token_chars = set(token)

    if token in ['<s>', '</s>', '<unk>']:
        return best

    for c in candidates:
        d = editdistance.eval(token, c)

        if d > min_distance:
            continue

        o = len(token_chars & set(c))
        if d < min_distance or o > max_overlap:
            best, min_distance, max_overlap = c, d, o

    return best


def match_vocab(vocab, vec_vocab, complete=False):
    matches = []
    for token in tqdm(vocab):
        m = None
        if token in vec_vocab:
            m = token
        elif token.lower() in vec_vocab:
            m = token.lower()
        elif token[0] == 'Ġ' and token[1:] in vec_vocab:
            m = token[1:]
        elif token[0] == 'Ġ' and token[1:].lower() in vec_vocab:
            m = token[1:].lower()
        elif complete:
            candidates = [t for t in vec_vocab if len(set(t) & set(token)) > 0]
            m = minimum_editdistance(token, candidates)

        matches.append(m)
    return matches


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('--model', required=False, default=None)
    args = parser.parse_args()

    print(f' > loading vectors for {args.lang}')
    ftx_path = Path('data') / args.lang / 'vectors' / 'fasttext' / f'wiki.{args.lang[:2]}.align.vec'
    ftx_cache_path = ftx_path.parent / f'{ftx_path.name}.pkl'

    if ftx_cache_path.exists():
        print(f' > loading {ftx_cache_path}')
        with open(ftx_cache_path, 'rb') as f:
            ftx_vecmap = pickle.load(f)
    else:
        ftx_vecmap = load_fasttext_vectormap(ftx_path)
        print(f' > saving {ftx_cache_path}')
        with open(ftx_cache_path, 'wb') as f:
            pickle.dump(ftx_vecmap, f)

    print(f' > loaded {len(ftx_vecmap):,} vectors')

    gpt_vocab = load_vocab(args.lang)
    matches = match_vocab(gpt_vocab, ftx_vecmap.keys(), complete=args.model is None)
    assert len(matches) == len(gpt_vocab)
    print(f' > finding matches for vocabulary size {len(gpt_vocab):,}')

    n_matches = sum(1 for t in matches if t is not None)
    print(f' > found {n_matches:,}/{len(gpt_vocab):,} vocab matches')
    assert args.model is not None or n_matches == len(matches)

    out_path = Path('data') / args.lang / 'vectors' / 'ftx.npy'
    ftx_embeds = [ftx_vecmap[m] for m in matches if m is not None]
    ftx_embeds = np.vstack(ftx_embeds)
    print(f' > exporting fasttext vectors with shape {ftx_embeds.shape} to {out_path}')
    np.save(out_path, ftx_embeds)

    if args.model is not None:
        print(f' > loading {args.model} embeddings')
        gpt_embeds = load_embeddings(args.model)
        assert len(gpt_embeds) == len(matches)
        gpt_embeds = [e for m, e in zip(matches, gpt_embeds) if m is not None]
        gpt_embeds = np.vstack(gpt_embeds)

        out_path = Path('data') / args.lang / 'vectors' / f'{args.model}.npy'
        print(f' > exporting {args.model} vectors with shape {gpt_embeds.shape} to {out_path}')
        np.save(out_path, gpt_embeds)


if __name__ == '__main__':
    main()
