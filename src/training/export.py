#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from .main import EmbeddingTunerModel


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('run', help='input run name from /runs')
    parser.add_argument('name', help='output target name for /models')
    parser.add_argument('--ckpt', default='last.ckpt')
    args = parser.parse_args()

    run_path = Path('data') / args.lang / 'runs' / args.run
    ckpt_path = run_path / 'checkpoints' / args.ckpt
    model_path = Path('data') / args.lang / 'models' / args.name

    if not run_path.exists():
        print(f'run path {run_path} does not exist')
        exit(1)
    if not ckpt_path.exists():
        print(f'checkpoint path {ckpt_path} does not exist')
        exit(1)
    if model_path.exists():
        print(f'output model path {model_path} already exists')
        exit(1)

    print(f'Loading checkpoint from {ckpt_path}')
    pl_model = EmbeddingTunerModel.load_from_checkpoint(str(ckpt_path))
    pl_model.m.save_pretrained(model_path)

    # Link to tokenizer.json
    # tokenizer_path = pl_model.hparams.tokenizer_path
    # os.symlink(os.path.relpath(tokenizer_path, model_path), model_path / 'tokenizer.json')

    print(f'Exported checkpoint {ckpt_path} to {model_path}')


if __name__ == '__main__':
    main()
