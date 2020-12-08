from argparse import ArgumentParser
import pickle
import os
from collections import Counter
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('lang')
parser.add_argument('--size',
                    type=int,
                    default=None,
                    help='vocab size (in thousands)')
args = parser.parse_args()

base_path = Path('data') / args.lang / 'preparation' / 'prepared'
data_paths = [base_path / 'data-{}k.pkl'.format(str(args.size).zfill(3))
              ] if args.size is not None else sorted(
                  base_path.glob('data*.pkl'))

for data_path in data_paths:
    data_path = str(data_path)
    print(f' > {data_path}')

    count_lengths = True
    count_coverages = True

    lengths_path = data_path + '.lengths'
    if os.path.exists(lengths_path):
        print(f' ::: {lengths_path} already exists. skipping')
        count_lengths = False

    coverage_path = data_path + '.coverage'
    if os.path.exists(coverage_path):
        print(f' ::: {coverage_path} already exists. skipping')
        count_coverages = False

    if not count_lengths and not count_coverages:
        continue

    print(' ::: loading examples')
    with open(data_path, 'rb') as f:
        examples = pickle.load(f)

    if count_lengths:
        print(' ::: counting lengths')
        lengths = [len(e) for e in examples]

        with open(lengths_path, 'wb') as f:
            pickle.dump(lengths, f)
        print(f' ::: saved {len(lengths)} lengths to {lengths_path}')

    if count_coverages:
        print(' ::: counting coverages')
        coverage = Counter()
        for e in examples:
            coverage.update(e)
        coverage = dict(coverage)

        with open(coverage_path, 'wb') as f:
            pickle.dump(coverage, f)
        print(f' ::: saved {len(coverage)} coverage scores to {coverage_path}')

    del examples
