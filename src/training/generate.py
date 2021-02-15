from argparse import ArgumentParser
import os
import json
from pathlib import Path
from time import time

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer


def gen(tokenizer_tgt: Tokenizer,
        model: GPT2LMHeadModel,
        device,
        prompt=None,
        n=10,
        tokenizer_eng=None,
        token_id_map=[],
        cfg={}):
    input_ids = None
    if prompt is not None and prompt.strip() != '':
        prompt = prompt.strip()
        if type(tokenizer_tgt) == Tokenizer:
            ids = [model.config.bos_token_id] + tokenizer_tgt.encode(
                prompt, None).ids
        else:
            ids = tokenizer_tgt.encode(prompt)
        input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)

    for _ in range(max(n // 5, 1)):
        m = min(5, n)

        batch_ids = model.generate(input_ids=input_ids,
                                   num_return_sequences=m,
                                   max_length=200,
                                   do_sample=True,
                                   top_k=10,
                                   top_p=0.9,
                                   temperature=2.0,
                                   repetition_penalty=10.0,
                                   num_beams=10,
                                   pad_token_id=cfg['pad_token_id'],
                                   bos_token_id=cfg['bos_token_id'],
                                   eos_token_id=cfg['eos_token_id'],
                                   no_repeat_ngram_size=4)

        for i in range(m):
            ids_tgt = batch_ids[i].flatten().tolist()
            txt_tgt = tokenizer_tgt.decode(ids_tgt,
                                           skip_special_tokens=True).strip()
            if tokenizer_eng is not None:
                ids_eng = [token_id_map[i] for i in ids_tgt if i not in [1, 2]]
                txt_eng = tokenizer_eng.decode(
                    ids_eng, skip_special_tokens=True).strip()
                yield txt_tgt, txt_eng
                continue
            yield txt_tgt


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('model')
    parser.add_argument('-n', type=int, default=None)
    args = parser.parse_args()

    with open(Path('data') / args.lang / 'config.json') as f:
        cfg = json.load(f)

    model_path = Path('data') / args.lang / 'models' / args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    os.environ['TOKENIZERS_PARALLELISM'] = str(False)
    # tokenizer_tgt = Tokenizer.from_file('tgt.tokenizer.json')
    if args.lang == 'ita':
        tokenizer_tgt = GPT2Tokenizer.from_pretrained(
            'LorenzoDeMattei/GePpeTto')
    else:
        tokenizer_tgt = Tokenizer.from_file(
            str(
                Path('data') / args.lang / 'preparation' / 'vocabularies' /
                'tokenizer.json'))

    # model: GPT2LMHeadModel = EmbeddingTunerModel.load_from_checkpoint(model_path).m
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(device)

    if args.n is not None:
        tokenizer_eng = GPT2Tokenizer.from_pretrained('gpt2')

        dict_path = Path(
            'data') / args.lang / 'dictionaries' / f'{args.model}.tsv'
        with open(dict_path) as f_map:
            token_id_map = [
                tokenizer_eng.convert_tokens_to_ids(
                    line.strip().split('\t')[1]) for line in f_map
            ]

        print(f'generating {args.n:,} random texts (unconditioned)')

        out_dir = Path('data') / args.lang / 'results' / 'examples'
        os.makedirs(out_dir, exist_ok=True)
        name = str(int(time()))

        tgt_out_path = out_dir / f'{name}.{args.lang}.txt'
        src_out_path = out_dir / f'{name}.eng.txt'

        print(
            f'generating {args.n} {args.lang} examples to {tgt_out_path} [{src_out_path}]'
        )
        with open(tgt_out_path, 'w') as f_tgt, open(src_out_path,
                                                    'w') as f_eng:
            for i, (tgt, eng) in enumerate(
                    gen(tokenizer_tgt,
                        model,
                        device,
                        n=args.n,
                        tokenizer_eng=tokenizer_eng,
                        token_id_map=token_id_map,
                        cfg=cfg)):
                print(f'{i:,}/{args.n:,}')
                f_tgt.write(tgt + '\n\n')
                f_eng.write(eng + '\n\n')

        return

    while True:
        print('\n##########################################')
        prompt = input(' > ').strip()

        for txt in gen(tokenizer_tgt, model, device, prompt, cfg=cfg):
            print('\n' + txt)


if __name__ == '__main__':
    main()
