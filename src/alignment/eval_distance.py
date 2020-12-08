from argparse import ArgumentParser
from pathlib import Path
import os
import json

from sklearn.metrics.pairwise import cosine_distances
import scipy

from .utils import get_model_size, load_embeddings, get_model_path


def get_embedding_similarity(lang, model_a, model_b, metric):
    a = load_embeddings(get_model_path(lang, model_a))
    b = load_embeddings(get_model_path(lang, model_b))
    assert a.shape == b.shape
    print(f'shape: {a.shape}')

    if metric == 'cosine':
        sim = cosine_distances(a, b).diagonal()
        return sim.mean()
    elif metric == 'euclidean':
        d = 0.
        for i in range(len(a)):
            d += scipy.spatial.distance.euclidean(a[i], b[i])
        d /= len(a)

        return d
    else:
        raise ValueError(f'invalid metric {metric}')


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('model_a')
    parser.add_argument('model_b', nargs='+')
    parser.add_argument('--metric',
                        default='euclidean',
                        choices=['euclidean', 'cosine'])
    args = parser.parse_args()

    for model_b in args.model_b:
        print(f'\n{args.model_a} vs. {model_b}\n')
        size_a = get_model_size(args.model_a)
        size_b = get_model_size(model_b)
        if size_a != size_b:
            print(f'models must have the same size. ({size_a} != {size_b})')
            continue

        res_path = Path(
            'data'
        ) / args.lang / 'results' / 'data' / 'distances' / f'{args.metric}.json'
        os.makedirs(res_path.parent, exist_ok=True)

        res = {}
        if os.path.exists(res_path):
            with open(res_path) as f:
                res = json.load(f)

        similarity = get_embedding_similarity(args.lang,
                                              args.model_a,
                                              model_b,
                                              metric=args.metric)
        print()
        print(f'{similarity:.3f}')

        mid = f'{args.model_a}@{model_b}'
        res[mid] = similarity

        with open(res_path, 'w') as f:
            json.dump(res, f, indent=2)

        print(f'\nSaved to {res_path}')


if __name__ == '__main__':
    main()
