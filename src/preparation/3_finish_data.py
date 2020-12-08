from argparse import ArgumentParser
import pickle
from pathlib import Path
import os
import itertools
import numpy as np

parser = ArgumentParser()
parser.add_argument('lang')
parser.add_argument('--size',
                    type=int,
                    default=None,
                    help='vocab size (in thousands)')
parser.add_argument('--subset', default=None, help='train,test,valid')
args = parser.parse_args()

if (args.size is None) == (args.subset is None):
    print('provide either size or subset')
    exit(1)

src_dir = Path('data') / args.lang / 'preparation' / 'prepared'
dst_dir = Path('data') / args.lang / 'preparation' / 'final'
if not dst_dir.exists():
    os.makedirs(dst_dir)

n = str(args.size).zfill(3) + 'k' if args.subset is None else args.subset

src_data_path = src_dir / f'data-{n}.pkl'
dst_data_path = dst_dir / f'data-{n}.npy'
if not dst_data_path.exists():
    print(f' > loading {src_data_path}')
    with open(src_data_path, 'rb') as src_f:
        data = pickle.load(src_f)

    print(f' > exporting {dst_data_path}')
    data = np.fromiter(itertools.chain.from_iterable(data),
                       dtype='uint32' if args.size is not None
                       and args.size > 60 else 'uint16')
    np.save(dst_data_path, data, allow_pickle=False)
    print(data.shape, data.dtype)
# else:
#     print(f' > loading {dst_data_path}')
#     data = np.load(dst_data_path)
#     print(data.shape, data.dtype)

src_index_path = src_dir / f'data-{n}.pkl.lengths'
dst_index_path = dst_dir / f'index-{n}.npy'
if not dst_index_path.exists():
    print(f' > loading {src_index_path}')
    with open(src_index_path, 'rb') as src_f:
        idx = pickle.load(src_f)

    idx = np.cumsum(idx, dtype='uint32')
    np.save(dst_index_path, idx, allow_pickle=False)
