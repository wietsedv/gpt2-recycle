from argparse import ArgumentParser
from pathlib import Path
import os
import json

from transformers import GPT2LMHeadModel
import numpy as np
import torch
from tqdm import tqdm


def get_file_perplexity(model: GPT2LMHeadModel,
                        path: Path,
                        limit=None,
                        block_size=1024,
                        stride=512):
    print(f' > loading tokenized data from {path}')
    encodings = np.load(path)
    encodings = torch.as_tensor(encodings).unsqueeze(0).cuda()

    print(f' > data shape: {encodings.shape}')

    print(f' > calculating perplexity [{block_size=}, {stride=}]')
    lls = []
    i = 0
    pbar = tqdm(range(1, encodings.size(1), stride))
    for i in pbar:
        begin_loc = max(i + stride - block_size, 0)
        end_loc = i + stride
        input_ids = encodings[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

        if limit is not None and len(lls) >= limit:
            break

        if len(lls) % 50 == 0:
            ppl = torch.exp(torch.stack(lls).sum() / i)
            pbar.set_description(f'PPL: {ppl:.3f}')

    if len(lls) == 0:
        return -1
    ppl = torch.exp(torch.stack(lls).sum() / i)
    return ppl


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('models', nargs='+')
    parser.add_argument('--subset', default='small')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('--limit', type=int, default=150_000)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--genres', nargs='+', default=None)
    args = parser.parse_args()

    src_path = Path(
        'data') / args.lang / 'evaluation' / 'tokenized' / args.subset

    for model_name in args.models:
        print(f'starting {model_name}')
        model_path = Path('data') / args.lang / 'models' / model_name

        if not src_path.exists():
            print(f' > source path {src_path} does not exist')
            exit(1)
        if not model_path.exists():
            print(
                f' > model path {model_path} does not exist. trying huggingface shortcut'
            )
            model_path = model_name

        print(f' > loading model {model_name}')
        model = GPT2LMHeadModel.from_pretrained(model_path).cuda()
        model.eval()

        pp_path = Path(
            'data'
        ) / args.lang / 'results' / 'data' / f'perplexities-{args.block_size}-{args.stride}' / args.subset / f'{model_name.replace("/", "_")}.json'
        os.makedirs(pp_path.parent, exist_ok=True)

        print(f' > saving perplexities to {pp_path}')

        res = {}
        if pp_path.exists():
            with open(pp_path, 'r') as f:
                res = json.load(f)

        for cat_path in sorted(src_path.glob('**/*.npy')):
            cat = cat_path.name.replace('.npy', '')
            if args.genres is not None and cat not in args.genres:
                continue

            print(f' > processing category "{cat}"')
            if cat in res:
                print(
                    f' > perplexity of "{cat}" is already known: {res[cat]:.3f}. skipping'
                )
                continue

            ppl = float(
                get_file_perplexity(model,
                                    cat_path,
                                    limit=args.limit,
                                    block_size=args.block_size,
                                    stride=args.stride))
            print(f' > PPL={ppl:.3f}')

            res[cat] = ppl
            with open(pp_path, 'w') as f:
                json.dump(res, f, indent=2)

        print(f'\n#### RESULTS for {args.lang} model {model_name} ####\n')
        for cat in sorted(res):
            print(f'{cat:<42} {res[cat]:.2f}')
        print('\n#####################\n\n')


if __name__ == '__main__':
    main()
