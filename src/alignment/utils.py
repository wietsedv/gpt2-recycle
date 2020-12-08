import os
from pathlib import Path, PosixPath

import numpy as np
import scipy as sp

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tokenizers import Tokenizer


_embedding_cache = {}

SIZE_MAP = {'LorenzoDeMattei_GePpeTto': 'sml'}
VECTOR_MODELS = {'ftx'}
SRC_MODELS = {'sml': 'gpt2', 'med': 'gpt2-medium', 'lrg': 'gpt2-large', 'xlg': 'gpt2-xl'}
LANG_TOKENIZERS = {'eng': 'gpt2', 'ita': 'LorenzoDeMattei/GePpeTto'}


def get_model_size(model):
    if model in SIZE_MAP:
        model_size = SIZE_MAP[model]
    else:
        model_size = model.split('.')[-1].split('_')[0]
    if model_size not in SRC_MODELS and model_size not in VECTOR_MODELS:
        print(f'unknown model size "{model_size}" in model "{model}"')
        exit(1)
    return model_size


def get_model_path(lang, model):
    if model in VECTOR_MODELS:
        return model
    if lang == 'eng':
        return model
    return Path('data') / lang / 'models' / model


def load_vocab(lang, model=''):
    if lang in LANG_TOKENIZERS:
        tokenizer = GPT2TokenizerFast.from_pretrained(LANG_TOKENIZERS[lang])
    else:
        tok_path = Path(model) / 'tokenizer.json'
        if not tok_path.exists():
            tok_path = Path('data') / lang / 'vocabularies' / 'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(tok_path))

    vocab = tokenizer.get_vocab()
    return sorted(vocab, key=vocab.get)


def load_model(model) -> GPT2LMHeadModel:
    if model in SRC_MODELS:
        model = SRC_MODELS[model]
    return GPT2LMHeadModel.from_pretrained(str(model))


def load_embeddings(path, lang=None, ftx=False):
    if path in VECTOR_MODELS or ftx:
        print(' > loading vectors with type', path)
        return np.load(Path('data') / lang / 'vectors' / f'{path}.npy')

    if path not in _embedding_cache:
        print(' > loading embeddings from model', path)
        model = load_model(path)
        _embedding_cache[path] = model.transformer.wte.weight.detach().numpy()
    return _embedding_cache[path]


def get_distances(src, tgt, path, dist_only=False, index_only=False, force=False, top_k=2000):
    d_path, i_path = None, None
    if not os.path.exists(Path(path).parent):
        os.makedirs(Path(path).parent)

    if dist_only and index_only:
        raise ValueError('dist_only and index_only cannot both be true')

    path = str(path)
    if not path.endswith('.npy'):
        raise ValueError('path must end with .npy extension')
    d_path = path
    i_path = path.replace('.npy', '.ind.npy')

    dist = None
    if not index_only or not os.path.exists(i_path) or force:
        if os.path.exists(d_path) and not force:
            dist = np.load(d_path)
        else:
            if type(src) in (str, PosixPath):
                src = load_embeddings(src)
            if type(tgt) in (str, PosixPath):
                tgt = load_embeddings(tgt)
            print(f' ::: calculating distances: {src.shape} {tgt.shape} (will be saved to {path})')
            dist = np.transpose(sp.spatial.distance.cdist(src, tgt, metric='euclidean'))
            np.save(d_path, dist)

        if dist_only:
            return dist

    ind = None
    if os.path.exists(i_path) and not force:
        ind = np.load(i_path)
    else:
        print(' ::: sorting pairwise distances')
        ind = np.argsort(dist, axis=1)[:, :top_k]
        np.save(i_path, ind)

    if index_only:
        return ind
    return dist, ind
