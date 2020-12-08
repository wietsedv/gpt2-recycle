from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from torch import randperm

parser = ArgumentParser()
parser.add_argument('lang')
parser.add_argument('size', type=int, help='vocab size (in thousands)')
parser.add_argument('val-ratio', type=float)
args = parser.parse_args()

n = args.size.zfill(3)

prep_dir = Path('data') / args.lang / 'preparation' / 'final'

src_path = prep_dir / f'index-{n}k.npy'
tra_dst_path = prep_dir / f'index-train-{n}k.npy'
val_dst_path = prep_dir / f'index-valid-{n}k.npy'

dat = np.load(src_path)

n_val = int(len(dat) * args.val_ratio)
n_tra = len(dat) - n_val

print(f'train={n_tra:,} valid={n_val:,}')

ind = randperm(len(dat)).tolist()
ind_tra = ind[:n_tra]
ind_val = ind[n_tra:]

np.save(tra_dst_path, ind_tra, allow_pickle=False)
np.save(val_dst_path, ind_val, allow_pickle=False)
