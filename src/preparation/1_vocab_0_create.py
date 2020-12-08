from argparse import ArgumentParser
from pathlib import Path

from tokenizers import Tokenizer, trainers, normalizers, pre_tokenizers, decoders, processors, models


def train_tokenizer(lang, vocab_size):
    with open(Path('data') / lang / 'preparation' / 'charset.txt') as f:
        alphabet = sorted([char[0] for char in f if char[0] != ' '])

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=['<unk>', '<s>', '</s>'],
        initial_alphabet=alphabet,
    )

    tokenizer.train(trainer, (Path('data') / lang / 'plaintext').glob('*/*.txt'))
    return tokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument('lang')
    parser.add_argument('size', type=int, help='vocab size (in thousands)')
    args = parser.parse_args()

    base_dir = Path('data') / args.lang / 'preparation' / 'vocabularies'

    dst_path = base_dir / f'{args.lang}-{str(args.size).zfill(3)}k.tokenizer.json'
    if dst_path.exists():
        print(f' > {dst_path} already exists. skipping')
        return

    print(f' > creating vocabulary with vocab size {args.size}k')
    tokenizer = train_tokenizer(args.lang, args.size * 1000)
    tokenizer.save(str(dst_path), pretty=True)


if __name__ == '__main__':
    main()
