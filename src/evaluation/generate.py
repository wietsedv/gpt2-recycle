from argparse import ArgumentParser
from pathlib import Path
import os

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


def generate(token_ids, model, tokenizer):
    input_ids = torch.LongTensor(token_ids).unsqueeze(0).cuda()

    batch_ids = model.generate(input_ids=input_ids,
                               num_return_sequences=1,
                               max_length=100,
                               do_sample=True,
                               top_k=20,
                               top_p=0.9,
                               temperature=2.0,
                               repetition_penalty=10,
                               num_beams=10,
                               no_repeat_ngram_size=4)

    ids = batch_ids[0].flatten().tolist()
    txt = f'[{tokenizer.decode(token_ids).strip()}]{tokenizer.decode(ids[len(token_ids):], skip_special_tokens=True)}'.replace(
        '\n\n', '\n')
    # txt = tokenizer.decode(ids, skip_special_tokens=True).strip()

    tqdm.write(f' > {txt}\n')
    return txt + '\n\n'


def main():
    parser = ArgumentParser()
    parser.add_argument('lang', choices=['nld', 'ita'])
    parser.add_argument('models', nargs='+')
    parser.add_argument('--src', default='small', choices=['full', 'small'])
    parser.add_argument('--file', default='full')
    parser.add_argument('-n', default=5, type=int)
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    base_path = Path(
        'data') / args.lang / 'evaluation' / 'examples' / args.src / args.file

    src_path = base_path / 'gold.txt'
    if not src_path.exists():
        print(f' > gold path {src_path} does not exist')
        exit(1)

    print(' > loading tokenizer')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if args.lang == 'ita':
        tokenizer = GPT2TokenizerFast.from_pretrained(
            'LorenzoDeMattei/GePpeTto')
    else:
        tokenizer_path = Path(
            'data') / args.lang / 'vocabularies' / 'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        args.n += 1

    print(f' > loading examples from {src_path}')
    examples = []
    with open(src_path) as f:
        for line in f:
            token_ids = tokenizer.encode(line.strip())
            if type(token_ids) != list:
                token_ids = [0] + token_ids.ids
            examples.append(token_ids[:args.n])
    print(f' > loaded {len(examples)} examples')

    for model_name in args.models:
        tgt_path = base_path / f'{model_name.replace("/", "_")}.txt'
        if not args.force and tgt_path.exists():
            print(f'{tgt_path} already exists. skipping')
            continue

        model_path = Path('data') / args.lang / 'models' / model_name
        if not model_path.exists():
            model_path = model_name

        print(f' > loading model {model_path}')
        model = GPT2LMHeadModel.from_pretrained(model_path).cuda()
        model.eval()

        print(' > generating endings for examples')
        generated = [
            generate(input_ids, model, tokenizer)
            for input_ids in tqdm(examples, ncols=80)
        ]
        with open(tgt_path, 'w') as f:
            f.writelines(generated)

        print(f'\nsaved to {tgt_path}')


if __name__ == '__main__':
    main()
