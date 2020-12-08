from argparse import ArgumentParser
from pathlib import Path
import os
import json

from tabulate import tabulate

from .utils import sort_models


def tabulate_distances(data, metric, fmt):
    table = []

    for m1 in sort_models(data):
        for m2 in sort_models(data[m1]):
            table.append((m1, m2, data[m1][m2]))

    return tabulate(table,
                    headers=['model 1', 'model 2', metric],
                    tablefmt=fmt,
                    floatfmt='.2f'
                    )


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('--fmt', default='latex')
    args = parser.parse_args()

    dist_path = Path('data') / args.lang / 'results' / 'data' / 'distances'

    for metric_path in dist_path.glob('*.json'):
        metric = metric_path.name.replace('.json', '')

        with open(metric_path) as f:
            data = json.load(f)

        grouped = {}
        for name, d in data.items():
            m1, m2 = name.split('@')
            if m1 not in grouped:
                grouped[m1] = {}
            grouped[m1][m2] = d

        tab = tabulate_distances(grouped, metric, args.fmt)

        if args.fmt == 'latex':
            out_path = Path('data') / args.lang / 'results' / 'tables' / 'distances' / f'{metric}.tex'
            os.makedirs(out_path.parent, exist_ok=True)

            with open(out_path, 'w') as f:
                f.write(tab)
            print(f'\nSaved table to {out_path}')


if __name__ == '__main__':
    main()
