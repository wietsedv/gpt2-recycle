from argparse import ArgumentParser
import json
import logging
from pathlib import Path

import scipy
import numpy as np
import torch
from torch import nn

from .utils import get_model_size, get_model_path, load_model, load_embeddings, get_distances
from .export_align_dictionary import SRC_BLACKLIST

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def ort_proc_transformation(src_a, src_b, tgt):
    print(
        f' > performing orthogonal Procrustes transformation ({src_a.shape} to {src_b.shape})'
    )

    padding = ((0, 0), (0, src_b.shape[1] - src_a.shape[1]))
    src_a = np.pad(src_a, padding)
    tgt = np.pad(tgt, padding)

    R, _ = scipy.linalg.orthogonal_procrustes(src_a, src_b)

    dst = tgt.dot(R)
    return dst


def lstsq_transformation(src_a, src_b, tgt):
    print(
        f' > performing least-squares procrustes transformation ({src_a.shape} to {src_b.shape})'
    )
    x, resids, _, _ = np.linalg.lstsq(src_a, src_b, rcond=None)
    print(f' ::: fitted: shape={x.shape} avg_residuals={resids.mean():.3f}')

    dst = np.dot(tgt, x)
    return dst


def knn_transformation(src_a,
                       src_b,
                       tgt,
                       dist_path,
                       k: int,
                       use_blacklist=False):
    print(
        f' > performing knn distance-weighted sum transformation ({src_a.shape} to {src_b.shape})'
    )
    dist, ind = get_distances(src_a, tgt, path=dist_path)

    if use_blacklist:
        ind2 = np.zeros((len(ind), k), dtype=np.int)
        for i in range(ind.shape[0]):
            j2 = 0
            for j in range(ind.shape[1]):
                if ind[i, j] in SRC_BLACKLIST:
                    continue
                ind2[i, j2] = ind[i, j]
                j2 += 1
                if j2 == k:
                    break
        ind = ind2
    else:
        ind = ind[:, :k]

    print(' ::: transforming tgt to dst')
    dst = np.zeros((tgt.shape[0], src_b.shape[1]), dtype=src_b.dtype)
    for i in range(len(dst)):
        embs = src_b[ind[i]]
        weights = 1 / dist[i, ind[i]]
        dst[i] = np.average(embs, axis=0, weights=weights)
    return dst


def embedding_to_model(lang, emb, base_model):
    model = load_model(base_model)
    model.config.vocab_size = emb.shape[0]

    with open(Path('data') / lang / 'config.json') as f:
        cfg = json.load(f)

    emb = torch.from_numpy(emb)

    model.config.pad_token_id = cfg['pad_token_id']
    model.config.bos_token_id = cfg['bos_token_id']
    model.config.eos_token_id = cfg['eos_token_id']
    model.transformer.wte.weight = nn.Parameter(emb)
    model.lm_head.weight = nn.Parameter(emb)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('src', help='Example: med')
    parser.add_argument('tgt', help='Example: sml_wte')
    parser.add_argument('-m',
                        '--method',
                        default='proc',
                        choices=['proc', 'lstsq', 'knn', 'bknn'])
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    size_a = get_model_size(args.tgt)
    size_b = get_model_size(args.src)
    mthd = args.method + (str(args.k)
                          if args.method in ['knn', 'bknn'] else '')
    print(
        f'Transforming from "{size_a}" to size "{size_b}" with method {mthd}')

    tgt_path = get_model_path(args.lang, args.tgt)
    dst_path = get_model_path(args.lang, f'{args.tgt}.{args.src}_{mthd}')
    if type(tgt_path) != str and not tgt_path.exists():
        print('model', args.tgt, 'does not exist')
        exit(1)

    dst = None
    if dst_path.exists() and not args.force:
        print('model', dst_path, 'already exists exist')
        exit(1)

    print(f'Destination will be: {dst_path}')

    src_a = load_embeddings(size_a, lang='eng')  # eng.ftx
    src_b = load_embeddings(size_b, lang='eng', ftx=size_a == 'ftx')  # eng.sml
    tgt = load_embeddings(tgt_path, lang=args.lang)  # nld.sml

    if args.method == 'proc':
        dst = ort_proc_transformation(src_a, src_b, tgt)
    elif args.method == 'lstsq':
        dst = lstsq_transformation(src_a, src_b, tgt)
    elif args.method in ['knn', 'bknn']:
        dist_path = Path('data') / args.lang / 'distances' / f'{args.tgt}.npy'
        dst = knn_transformation(src_a,
                                 src_b,
                                 tgt,
                                 dist_path,
                                 k=args.k,
                                 use_blacklist=args.method == 'bknn')
    else:
        raise NotImplementedError(f'unkown method {args.method}')

    print(' > exporting model')
    dst_model = embedding_to_model(args.lang, dst, size_b)
    dst_model.save_pretrained(dst_path)
    # dst_tok_path = dst_path / 'tokenizer.json'
    # if not dst_tok_path.exists():
    #     shutil.copyfile(tgt_path / 'tokenizer.json', dst_tok_path, follow_symlinks=False)
    print(' > saved to', str(dst_path) + '\n')


if __name__ == '__main__':
    main()
