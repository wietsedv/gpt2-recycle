from argparse import ArgumentParser
import json
from pathlib import Path
import os

from .export_align_dictionary import get_nearest_vocab_alignment


def load_dictionary(dict_path):
    dictionary = []
    with open(dict_path) as f:
        for line in f:
            tokens = line.rstrip().split('\t')
            dictionary.append((tokens[0], tokens[1:]))
    return dictionary


def get_dictionary_overlap(lang, model_a, model_b, ranks):
    max_rank = max(ranks)

    dict_dir = Path('data') / lang / 'dictionaries'
    dict_path_a = dict_dir / f'{model_a}.tsv'
    dict_path_b = dict_dir / f'{model_b}.tsv'

    if not dict_path_a.exists():
        print(f'{dict_path_a} does not exist. run \'$ align_vocab {model_a}\'')
        exit(1)
    if not dict_path_b.exists():
        print(f'{dict_path_b} does not exist. run \'$ align_vocab {model_b}\'')
        exit(1)

    print('loading dictionaries')
    dict_a = load_dictionary(dict_path_a)
    dict_b = load_dictionary(dict_path_b)
    assert len(dict_a) == len(dict_b)

    print('counting dictionary overlap')
    overlap = {k: 0 for k in ranks}

    n_min = max_rank

    for (t_a, src_a), (t_b, src_b) in zip(dict_a, dict_b):
        assert t_a == t_b

        if len(src_a) < n_min:
            n_min = len(src_a)
        if len(src_b) < n_min:
            n_min = len(src_b)

        for k in ranks:
            overlap[k] += len(set(src_a[:k]) & set(src_b[:k]))

    if n_min < max_rank:
        print(
            f'WARNING: dictionary contains rows with only {n_min} items, but the highest rank is {max_rank}'
        )

    overlap = {k: n / k / len(dict_a) for k, n in overlap.items()}
    return overlap


def get_nn_overlap(lang, model_a, model_b, ranks, match_space, use_blacklist,
                   ignore_chars):
    max_rank = max(ranks)

    print('aligning model a')
    alignments_a, _, _ = get_nearest_vocab_alignment(
        lang,
        model_a,
        n=max_rank,
        match_space=match_space,
        use_blacklist=use_blacklist,
        ignore_chars=ignore_chars)
    print('aligning model b')
    alignments_b, _, _ = get_nearest_vocab_alignment(
        lang,
        model_b,
        n=max_rank,
        match_space=match_space,
        use_blacklist=use_blacklist,
        ignore_chars=ignore_chars)
    assert len(alignments_a) == len(alignments_b)

    print('counting alignment overlap')
    overlap = {k: 0 for k in ranks}
    n_min = max_rank

    for (t_a, src_a), (t_b, src_b) in zip(alignments_a, alignments_b):
        assert t_a == t_b

        if len(src_a) < n_min:
            n_min = len(src_a)
        if len(src_b) < n_min:
            n_min = len(src_b)

        for k in ranks:
            overlap[k] += len(set(src_a[:k]) & set(src_b[:k]))

    if n_min < max_rank:
        print(
            f'WARNING: dictionary contains rows with only {n_min} items, but the highest rank is {max_rank}'
        )

    overlap = {k: n / k / len(alignments_a) for k, n in overlap.items()}
    return overlap


def get_overlap_path(args, model_b):
    argstr = ''
    if args.use_blacklist:
        argstr += '-blacklist'
    if args.match_space:
        argstr += '-matchspace'
    if args.ignore_chars:
        argstr += '-nochars'
    if len(argstr) > 0:
        argstr = '?' + argstr[1:]

    out_path = Path(
        'data'
    ) / args.lang / 'results' / 'data' / 'overlap' / f'{args.model_a}@{model_b}{argstr}.json'
    return out_path


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('model_a')
    parser.add_argument('model_b', nargs='+')
    parser.add_argument('--rank',
                        nargs='+',
                        default=[
                            1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140,
                            160, 180, 200, 250, 300, 350, 400, 450, 500, 550,
                            600, 650, 700, 750, 800, 850, 900, 950, 1000
                        ])
    parser.add_argument('--use-dict', action='store_true')
    parser.add_argument('--use-blacklist', action='store_true')
    parser.add_argument('--match-space', action='store_true')
    parser.add_argument('--ignore-chars', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    for model_b in args.model_b:
        out_path = get_overlap_path(args, model_b)
        if not args.force and out_path.exists():
            print(f'{out_path} already exists. use -f to override')
            continue

        if args.use_dict:
            overlap = get_dictionary_overlap(args.lang, args.model_a, model_b,
                                             args.rank)
        else:
            overlap = get_nn_overlap(args.lang,
                                     args.model_a,
                                     model_b,
                                     ranks=args.rank,
                                     match_space=args.match_space,
                                     use_blacklist=args.use_blacklist,
                                     ignore_chars=args.ignore_chars)

        print('\nAverage Overlap at K')
        for k, score in overlap.items():
            print(f'K={k:>3} {score:.3f}')

        # Plot
        os.makedirs(out_path.parent, exist_ok=True)
        # plot_scores(overlap, fig_path)
        with open(out_path, 'w') as f:
            json.dump(overlap, f, indent=2)
        print(f'\nScores saved to {out_path}')


if __name__ == '__main__':
    main()
