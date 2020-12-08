import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# dead code


def plot_scores(lang, model_name):
    scores_paths = (Path('data') / lang / 'models' / model_name / 'derived').glob('*/scores.json')

    for path in sorted(scores_paths):
        print(path)

        with open(path) as f:
            scores = json.load(f)
            scores = sorted([(int(rank), score) for rank, score in scores.items()])

            x = [s[0] for s in scores]
            y = [s[1] for s in scores]

            model_name = str(Path(path).parent.name)
            plt.plot(x, y, label=model_name)

    plt.title('Precision@n@n')
    plt.xlabel('n')
    plt.ylabel('precision@n')
    plt.legend()


def main():
    lang = 'nld'
    os.makedirs(Path('data') / lang / 'tmp' / 'plots', exist_ok=True)

    plot_scores(lang, 'sml_wte_32bit')
    plt.savefig(Path('data') / lang / 'tmp' / 'plots' / 'alignment_scores.png')
    plt.close()


if __name__ == '__main__':
    main()
