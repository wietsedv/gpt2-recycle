from argparse import ArgumentParser
import json
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('lang')
args = parser.parse_args()

with open(Path('data') / args.lang / 'preparation' / 'charset.txt') as f:
    charset = [char[0] for char in f if char[0] != ' ']

for vocab_path in sorted(
    (Path('data') / args.lang / 'preparation' /
     'vocabularies').glob(f'{args.lang}-*k.tokenizer.json')):
    with open(vocab_path) as f:
        vocab = json.load(f)['model']['vocab']

    print(f' > {vocab_path} [{len(vocab):>7,} tokens]')

    # check completeness
    ok = True
    for char in charset:
        if char not in vocab:
            print(f' ::: "{char}" is missing')
            ok = False
    if ok:
        print('all whitelisted characters are present!')

    print()
