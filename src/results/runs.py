from argparse import ArgumentParser
from pathlib import Path
import os

import matplotlib.pyplot as plt

RUNS = {
    'nld': {
        'slow_vs_unfreeze': {
            'sml_fullslow': [13477410, 13507945],
            'sml_unfreeze': [13478174, 13517039]
        },
    },
    'ita': {}
}

# def plot_all_scores(lang, overlaps, name):
#     basepath = Path('data') / lang / 'results' / 'figures' / 'runs'
#     os.makedirs(basepath, exist_ok=True)

#     for m1 in sort_models(overlaps.keys()):
#         path = basepath / f'{m1}?{name}.png'

#         plt.title(f'Average Overlap with {m1} at top K tokens')
#         plt.xlabel('K')
#         plt.ylabel(f'Average Overlap with {m1}')

#         for m2 in sort_models(overlaps[m1].keys()):
#             ranks = sorted(overlaps[m1][m2].keys())
#             scores = [overlaps[m1][m2][k] for k in ranks]
#             plt.plot(ranks, scores, label=m2)

#         plt.legend()
#         plt.savefig(path)
#         plt.close()
#         print(f'Saved figure to {path}')


def plot_values(lang, run_values, title):
    path = Path(
        'data') / lang / 'results' / 'figures' / 'runs' / f'{lang}_{title}.png'
    os.makedirs(path.parent, exist_ok=True)

    plt.xlabel('gpu hours')
    plt.ylabel('loss')

    for name, values in run_values:
        time = [v[0] / 3600 * 8 for v in values]
        loss = [v[2] for v in values]
        plt.plot(time, loss, label=name)

    # plt.legend()
    plt.savefig(path, bbox_inches='tight')
    print(f'saved to {path}\n')
    plt.close()


def read_csv(runs_dir, run):
    print(f'reading {run} from {runs_dir}')
    run_path = list(runs_dir.glob(f'run-{run}-*.csv'))[0]

    values = []

    with open(run_path) as f:
        next(f)

        for line in f:
            w, s, v = line.rstrip().split(',')
            values.append((float(w), s, float(v)))

    return values


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    args = parser.parse_args()

    runs_dir = Path('data') / args.lang / 'results' / 'data' / 'runs'
    if not runs_dir.exists():
        print(f'{runs_dir} does not exist')
        exit(1)

    for title, runsets in RUNS[args.lang].items():
        run_values = []
        for name, runs in runsets.items():
            print(name, len(runs))

            values, offset = [], 0
            for run in runs:
                new_values = read_csv(runs_dir, run)
                t0 = new_values[0][0]

                new_values = [((w - t0) + offset, s, v)
                              for w, s, v in new_values]
                values.extend(new_values)
                offset = new_values[-1][0] + (new_values[-1][0] -
                                              new_values[-2][0])
            run_values.append((name, values))

        plot_values(args.lang, run_values, title)


if __name__ == '__main__':
    main()
