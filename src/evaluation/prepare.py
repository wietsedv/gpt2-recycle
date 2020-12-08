from argparse import ArgumentParser
from pathlib import Path
import os
from xml.etree.ElementTree import ElementTree

from transformers import GPT2TokenizerFast
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np


def parse_sonar_xml(path, md):
    tree = ElementTree()
    with open(path, encoding='iso-8859-15') as src_f:
        tree.parse(src_f)

    ns = '{http://lands.let.ru.nl/projects/d-coi/ns/1.0}'

    lines = []
    content = tree.find(f'./{ns}text/{ns}body')

    for sec in content:
        for p in sec:
            tokens = [w.text for s in p for w in s]
            lines.append(md.detokenize(tokens))

    doc = '\n'.join(lines) + '\n'
    return doc


def convert_sonar(src_dir, dst_dir, whitelist=None, include_full=False, force=False):
    from sacremoses import MosesDetokenizer

    md = MosesDetokenizer(lang='nl')

    doc_names = set()

    full_path = dst_dir / 'full.txt'
    full_dst_f = None
    if include_full and not full_path.exists():
        force = True
        full_dst_f = open(full_path, 'w')

    categories = sorted([d for d in os.listdir(src_dir) if os.path.isdir(src_dir / d)])
    print(f'> converting {len(categories):,} categories in {src_dir}\n')

    for i, cat in enumerate(categories, start=1):
        dst_path = dst_dir / f'{cat}.txt'
        print(f'[{i:>3,}/{len(categories):,}] {cat} ({dst_path})')

        if dst_path.exists() and not force:
            print(f' > destination path {dst_path} already exists. skipping')
            continue

        xml_paths = [p for p in sorted((src_dir / cat).glob('**/*.dcoi.xml')) if not p.name.startswith('.')]
        if whitelist is not None:
            xml_paths = [p for p in xml_paths if p.name.split('.')[0] in whitelist]
            doc_names.update([p.name.split('.')[0] for p in xml_paths])
        print(f' > {len(xml_paths):,} xml files')

        with open(dst_path, 'w') as dst_f:
            for j, xml_path in enumerate(tqdm(xml_paths, ncols=60), start=1):
                doc = parse_sonar_xml(xml_path, md)
                if full_dst_f is not None:
                    full_dst_f.write(doc + '\n')
                if j < len(xml_paths):
                    doc += '\n'
                dst_f.write(doc)

    if full_dst_f is not None:
        full_dst_f.close()

    print(f'converted {len(doc_names)} (whitelist={len(whitelist)})')


def tokenize_corpus(src_dir, dst_dir, tokenizer: Tokenizer, force):
    for i, doc_path in enumerate(sorted(src_dir.glob('*.txt')), start=1):
        cat = doc_path.name.replace('.txt', '')
        dst_path = dst_dir / f'{cat}.npy'
        print(f'[{i:>3,}] {cat} ({dst_path})')

        if dst_path.exists() and not force:
            print(f' > destination path {dst_path} already exists. skipping')
            continue

        token_ids = []

        print(f' > reading {doc_path}')
        if cat == 'full':
            n_lines = 500_000
            print(f'reading in chunks of {n_lines:,} lines')

            with open(doc_path) as f:
                lines = []
                for line in f:
                    lines.append(line)

                    if len(lines) >= n_lines:
                        print(f' > tokenizing {len(lines):,} lines')
                        token_ids.extend(tokenizer.encode(''.join(lines)).ids)
                        lines = []

            if len(lines) > 0:
                print(f' > tokenizing {len(lines):,}')
                token_ids.extend(tokenizer.encode(''.join(lines)).ids)

            token_ids = np.array(token_ids)
        else:
            with open(doc_path) as f:
                txt = f.read()

            print(' > tokenizing')
            token_ids = np.array(tokenizer.encode(txt).ids)

        print(f' > saving to {dst_path}')
        np.save(dst_path, token_ids)


# def tokenize_corpus(src_dir, dst_dir, tokenizer: Tokenizer, force):
#     def add_doc(lines, token_ids):
#         token_ids.extend([1] + tokenizer.encode(''.join(lines)).ids + [2])

#     for i, doc_path in enumerate(sorted(src_dir.glob('*.txt')), start=1):
#         cat = doc_path.name.replace('.txt', '')
#         dst_path = dst_dir / f'{cat}.npy'
#         print(f'[{i:>3,}] {cat} ({dst_path})')

#         token_ids = []

#         if dst_path.exists() and not force:
#             print(f' > destination path {dst_path} already exists. skipping')
#             continue

#         print(f' > reading {doc_path}')

#         with open(doc_path) as f:
#             lines = []
#             for line in f:
#                 if line == '\n':
#                     add_doc(lines, token_ids)
#                     lines = []
#                     continue
#                 lines.append(line)

#         if len(lines) > 0:
#             add_doc(lines, token_ids)

#         token_ids = np.array(token_ids)

#         print(f' > saving to {dst_path}')
#         np.save(dst_path, token_ids)


def list_sonar_files(path):
    filenames = set()
    subpaths = [
        path / 'COREF' / 'SONAR_1_COREF',
        path / 'NE' / 'SONAR_1_NE' / 'IOB',
        path / 'POS' / 'SONAR_1_POS',
        path / 'SPT' / 'SONAR_1_STEx'
    ]

    for p in subpaths:
        # filenames.update([f.split('.')[0] for f in os.listdir(p) if f[:3] in {'WS-', 'WR-'}])
        filenames.update([f.split('.')[0] for f in os.listdir(p)])

    return filenames


def prepare_nld(src_path, dst_path, tgt_path, tokenizer_path, force=False):
    sonar_src = src_path / 'sonar' / 'SONAR500' / 'DCOI'

    # print(' > Converting SONAR-500 to plaintext')
    # s500_plaintext = dst_path / 'full'
    # os.makedirs(s500_plaintext, exist_ok=True)
    # convert_sonar(sonar_src, s500_plaintext, include_full=True, force=force)

    print(' > Reading SONAR-1 files')
    s1_path = src_path / 'sonar' / 'SONAR1'
    s1_files = list_sonar_files(s1_path)
    print(f'{len(s1_files)} files')

    print('\n > Converting SONAR-1 to plaintext')
    s1_plaintext = dst_path / 'small'
    os.makedirs(s1_plaintext, exist_ok=True)
    convert_sonar(sonar_src, s1_plaintext, whitelist=s1_files, include_full=True, force=force)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # print('\n > Tokenizing SONAR-500')
    # s500_tokenized = tgt_path / 'full'
    # os.makedirs(s500_tokenized, exist_ok=True)
    # tokenize_corpus(s500_plaintext, s500_tokenized, tokenizer=tokenizer, force=force)

    print('\n > Tokenizing SONAR-1')
    s1_tokenized = tgt_path / 'small'
    os.makedirs(s1_tokenized, exist_ok=True)
    tokenize_corpus(s1_plaintext, s1_tokenized, tokenizer=tokenizer, force=force)


def tokenize_ita_file(tokenizer: GPT2TokenizerFast, src_path, eot_token='<|endoftext|>', min_length=30, eot=[0]):
    token_ids = []
    doc = ''
    with open(src_path) as f:
        for line in tqdm(f):
            if line == eot_token + '\n':
                if len(doc) >= min_length:
                    token_ids.extend(tokenizer.encode(doc.strip()) + eot)
                doc = ''
                continue

            doc += line

    if doc != '':
        token_ids.extend(tokenizer.encode(doc.strip()) + eot)
    return token_ids


def prepare_ita(src_path, dst_path, tgt_path, force=False):
    tokenizer = GPT2TokenizerFast.from_pretrained('LorenzoDeMattei/GePpeTto')

    # subsets = ['small']
    subsets = ['small', 'full']
    for subset in subsets:
        tgt_subset = subset
        # tgt_subset = f'{subset}-skipeot'
        os.makedirs(tgt_path / tgt_subset, exist_ok=True)

        print(src_path / subset)

        for src_file in (src_path / subset).glob('*.txt'):
            print(f' > Tokenizing data from {src_file}')

            tgt_file = tgt_path / tgt_subset / Path(src_file).name.replace('.txt', '.npy')
            if not force and tgt_file.exists():
                print(f'skipping. {tgt_file} already exists')
                continue

            token_ids = tokenize_ita_file(tokenizer, src_file)
            np.save(tgt_file, np.array(token_ids))
            print(f'Saved to {tgt_file}')


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    src_path = Path('data') / args.lang / 'evaluation' / 'sources'
    dst_path = Path('data') / args.lang / 'evaluation' / 'plaintext'
    tgt_path = Path('data') / args.lang / 'evaluation' / 'tokenized'

    if not src_path.exists():
        print(f'source path {src_path} does not exist')
        exit(1)

    os.makedirs(dst_path, exist_ok=True)

    if args.lang == 'nld':
        tokenizer_path = Path('data') / args.lang / 'vocabularies' / 'tokenizer.json'
        if not tokenizer_path.exists():
            print(f'tokenizer vocabulary path {tokenizer_path} does not exist')
            exit(1)
        return prepare_nld(src_path, dst_path, tgt_path, tokenizer_path, force=args.force)
    elif args.lang == 'ita':
        return prepare_ita(src_path, dst_path, tgt_path, force=args.force)

    print(f'unsupported language {args.lang}')
    exit(1)


if __name__ == '__main__':
    main()
