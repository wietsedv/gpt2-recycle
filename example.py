import os
from argparse import ArgumentParser
from transformers import pipeline

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='gpt2-small-dutch')
parser.add_argument('-d', '--device', type=int, default=-1)
args = parser.parse_args()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

p = pipeline('text-generation', model=args.model, device=args.device)

while True:
    txt = input('prompt > ')
    txt = p(txt)[0]['generated_text']
    print(txt)
