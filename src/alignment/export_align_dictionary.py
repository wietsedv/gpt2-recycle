from argparse import ArgumentParser
from pathlib import Path
import os
from collections import Counter

from .utils import get_model_size, get_model_path, get_distances, load_vocab


SRC_BLACKLIST = {30208, 30212, 23090, 42066, 42089, 124, 125, 9364, 153, 154, 33434, 155, 27293, 173, 174, 30897,
                 30898, 179, 180, 181, 178, 5815, 183, 184, 30905, 185, 186, 182, 187, 188, 189, 190, 192, 191, 194,
                 195, 37574, 193, 197, 196, 199, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
                 215, 216, 217, 218, 219, 17629, 221, 39142, 39655, 40219, 39714, 40240, 40241, 39749, 30899, 39752,
                 36173, 31573, 13150, 177, 27013, 39820, 34718, 15272, 45544, 14827, 200}


def align_vocabulary(ind, src_vocab, tgt_vocab, n, match_space, use_blacklist, ignore_chars):
    print(f'ind={len(ind)}, vocab={len(tgt_vocab)}')
    assert len(ind) >= len(tgt_vocab)
    if ind.shape[1] < n:
        print(
            f'distance index is not usable. alignment wants {n} items, but the index only contains top {ind.shape[1]}')
        exit(1)

    alignment = []
    for tgt_i in range(len(tgt_vocab)):
        tgt_token = tgt_vocab[tgt_i]

        if ignore_chars and (len(tgt_token) == 1 or (len(tgt_token) == 2 and tgt_token[0] == 'Ġ')):
            continue

        src_tokens = []
        for src_i in ind[tgt_i]:
            src_token = src_vocab[src_i]

            if ignore_chars and (len(src_token) == 1 or (len(src_token) == 2 and src_token[0] == 'Ġ')):
                continue

            if use_blacklist and src_i in SRC_BLACKLIST:
                continue

            if match_space and (tgt_token[0] == 'Ġ') != (src_token[0] == 'Ġ'):
                continue

            src_tokens.append(src_token)
            if len(src_tokens) == n:
                break

        alignment.append((tgt_token, src_tokens))

    return alignment


def get_nearest_vocab_alignment(lang, model, n, match_space=True, use_blacklist=True, ignore_chars=False):
    # Determine source and target model identifiers
    model_size = get_model_size(model)
    model_path = get_model_path(lang, model)

    # Load distances
    dist_path = Path('data') / lang / 'distances' / f'{model}.npy'
    ind = get_distances(model_size, model_path, path=dist_path, index_only=True)

    # Load vocabularies
    src_vocab = load_vocab('eng', model_size)  # Original (English) vocabulary
    tgt_vocab = load_vocab(lang, model_path)  # Target (non-English) vocabulary
    print(f'ind={len(ind)}, vocab={len(tgt_vocab)}')
    assert len(ind) >= len(tgt_vocab)

    # Align vocabularies
    alignment = align_vocabulary(ind, src_vocab, tgt_vocab, n, match_space, use_blacklist, ignore_chars)

    return alignment, src_vocab, tgt_vocab


def main():
    parser = ArgumentParser(description='For each target vocab token, find the closest source vocab tokens')
    parser.add_argument('lang')
    parser.add_argument('model', help='model name from /models')
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('--src-max', type=int, default=200)
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    suffix = ''
    if args.n is None:
        args.n = 20
    else:
        suffix = f'.{args.n}'

    # Output path
    dict_path = Path('data') / args.lang / 'dictionaries' / f'{args.model}{suffix}.tsv'
    if not dict_path.parent.exists():
        os.makedirs(dict_path.parent)
    if dict_path.exists() and not args.force:
        print(f'{dict_path} already exists. use -f to overwrite')
        exit(1)

    print(f'Blacklist size: {len(SRC_BLACKLIST):,}')

    alignment, src_vocab, _ = get_nearest_vocab_alignment(args.lang, args.model, args.n)

    with open(dict_path, 'w') as f, open(str(dict_path).replace('.tsv', '.words.tsv'), 'w') as f_words:
        for tgt_token, src_tokens in alignment:
            f.write(tgt_token + '\t' + '\t'.join(src_tokens) + '\n')
            if tgt_token[0] == 'Ġ' and len(tgt_token) > 2:
                f_words.write(tgt_token[1:] + '\t' + '\t'.join([t[1:] for t in src_tokens]) + '\n')

    print(f'\nDictionary is written to {dict_path}')

    # Source tokens that are too often closest to a target token
    blacklist_updated = False
    print('\nOverused source tokens: (repeat updating SRC_BLACKLIST until no new items are added)')
    src_counts = Counter([src_tokens[0] for _, src_tokens in alignment])
    for src_token, n in src_counts.most_common(100):
        if n < args.src_max:
            break
        src_i = src_vocab.index(src_token)
        print(f'{src_i:>10} {src_token:>20} {n:>5,}')
        SRC_BLACKLIST.add(src_i)
        blacklist_updated = True

    if blacklist_updated:
        print('SRC_BLACKLIST =', SRC_BLACKLIST)
    else:
        print(' > No overused src tokens')


if __name__ == '__main__':
    main()
