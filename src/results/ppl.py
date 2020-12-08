from argparse import ArgumentParser
from pathlib import Path
import os
import json

from tabulate import tabulate

from .utils import get_model_size, sort_models

# Subcorpora with < 5K lines
BLACKLIST = {
    'WR-P-E-E_newsletters', 'WR-P-E-H_teletext_pages', 'WR-P-E-K_blogs',
    'WR-P-P-D_newsletters'
}


def tabulate_perplexities(names, perps, fmt):
    table = {}

    for i, p in enumerate(perps, start=1):
        for cat, ppl in p.items():
            if ppl < 0 or cat in BLACKLIST:
                continue

            if cat not in table:
                table[cat] = [cat] + [None] * len(names)
            table[cat][i] = ppl

    return tabulate(table.values(),
                    headers=['genre'] + names,
                    tablefmt=fmt,
                    floatfmt='.2f'
                    #    floatfmt='.2e'
                    )


def tabulate_perplexities_short(names, perps, fmt):
    table = []

    for n, p in zip(names, perps):
        if 'full' not in p:
            continue
        table.append((n, p['full']))

    return tabulate(table,
                    headers=['model', 'ppl'],
                    tablefmt=fmt,
                    floatfmt='.2f'
                    #    floatfmt='.2e'
                    )


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('--subset', default='small')
    parser.add_argument('--fmt', default='simple')
    parser.add_argument('--size', default=None)
    parser.add_argument('--method', default=None)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    args = parser.parse_args()

    pp_dir = Path(
        'data'
    ) / args.lang / 'results' / 'data' / f'perplexities-{args.block_size}-{args.stride}' / args.subset
    if not pp_dir.exists():
        print(f'{pp_dir} does not exist')
        exit(1)

    model_names = [m.replace('.json', '') for m in os.listdir(pp_dir)]
    model_names = sort_models(model_names)

    if args.size is not None:
        model_names = [
            m for m in model_names if get_model_size(m) == args.size
        ]
    if args.method is not None:
        model_names = [
            m for m in model_names if args.method in m.split('_')[-1]
        ]

    model_perps = []
    for m in model_names:
        with open(pp_dir / f'{m}.json') as f:
            model_perps.append(json.load(f))

    if args.short:
        tab = tabulate_perplexities_short(model_names,
                                          model_perps,
                                          fmt=args.fmt)
    else:
        tab = tabulate_perplexities(model_names, model_perps, fmt=args.fmt)

    print(tab)

    if args.fmt == 'latex':
        name = f'{args.subset}'
        if args.short:
            name = f'{name}-short'
        out_path = Path(
            'data'
        ) / args.lang / 'results' / 'tables' / 'perplexities' / f'{name}.tex'
        os.makedirs(out_path.parent, exist_ok=True)

        with open(out_path, 'w') as f:
            f.write(tab)
        print(f'\nsaved table to {out_path}')
    else:
        print('\nWARNING: not saving results. only saving if --fmt latex')


if __name__ == '__main__':
    main()
