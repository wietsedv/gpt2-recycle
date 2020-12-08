from argparse import ArgumentParser
from pathlib import Path
import os
import json

import matplotlib.pyplot as plt
from tabulate import tabulate

from .utils import sort_models


def plot_all_scores(lang, overlaps, name):
    basepath = Path('data') / lang / 'results' / 'figures' / 'overlap'
    os.makedirs(basepath, exist_ok=True)

    for m1 in sort_models(overlaps.keys()):
        path = basepath / f'{m1}?{name}.png'

        plt.title(f'Average Overlap with {m1} at top K tokens')
        plt.xlabel('K')
        plt.ylabel(f'Average Overlap with {m1}')

        for m2 in sort_models(overlaps[m1].keys()):
            ranks = sorted(overlaps[m1][m2].keys())
            scores = [overlaps[m1][m2][k] for k in ranks]
            plt.plot(ranks, scores, label=m2)

        plt.legend()
        plt.savefig(path)
        plt.close()
        print(f' > saved figure to {path}')


def tabulate_overlap(overlaps, fmt, k=1000):
    table = []

    for m1 in sort_models(overlaps.keys()):
        for m2 in sort_models(overlaps[m1].keys()):
            if k not in overlaps[m1][m2]:
                continue
            score = overlaps[m1][m2][k]
            table.append((m1, m2, score))

    return tabulate(table,
                    headers=['model 1', 'model 2', f'overlap@{k}'],
                    tablefmt=fmt,
                    floatfmt='.2f')


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('--fmt', default='simple')
    args = parser.parse_args()

    overlap_dir = Path('data') / args.lang / 'results' / 'data' / 'overlap'
    if not overlap_dir.exists():
        print(f'{overlap_dir} does not exist')
        exit(1)

    overlaps = {}
    for filepath in overlap_dir.glob('*.json'):
        name = filepath.name.replace('.json', '')
        models, options = name.split('?') if '?' in name else (name, None)
        m1, m2 = models.split('@')
        options = tuple(options.split('-')) if options is not None else tuple()

        if options not in overlaps:
            overlaps[options] = {}
        if m1 not in overlaps[options]:
            overlaps[options][m1] = {}

        with open(filepath) as f:
            scores = json.load(f)
        scores = {int(k): v for k, v in scores.items() if int(k) > 10}
        overlaps[options][m1][m2] = scores

    for options in sorted(overlaps):
        print(f'\n### {options} ###')
        n = '-'.join(options)
        plot_all_scores(args.lang, overlaps[options], name=n)

        out_path = Path(
            'data') / args.lang / 'results' / 'tables' / 'overlap' / f'{n}.tex'
        os.makedirs(out_path.parent, exist_ok=True)

        tab = tabulate_overlap(overlaps[options], args.fmt)

        with open(out_path, 'w') as f:
            f.write(tab)
        print(f' > saved table to {out_path}\n')
        print(tab)


if __name__ == '__main__':
    main()
