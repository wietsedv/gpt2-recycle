import sys

from .export_align_dictionary import SRC_BLACKLIST
from .utils import load_vocab


def main():
    vocab = load_vocab(sys.argv[1], 'sml')
    for idx in sorted(SRC_BLACKLIST):
        print(vocab[idx])


if __name__ == '__main__':
    main()
