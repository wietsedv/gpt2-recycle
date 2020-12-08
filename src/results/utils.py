import re

SIZE_MAP = {'LorenzoDeMattei_GePpeTto': 'sml'}
SRC_MODELS = {
    'sml': 'gpt2',
    'med': 'gpt2-medium',
    'lrg': 'gpt2-large',
    'xlg': 'gpt2-xl'
}
MODEL_SIZES = ['sml', 'med', 'lrg', 'xlg']
# METHODS = ['GePpeTto', 'proc', 'lstsq', 'knn1', 'knn5', 'knn10', 'knn20', 'knn40', 'wte', 'full10', 'fullslow10']
METHODS = [
    'GePpeTto', 'proc', 'lstsq', 'knn', 'wte', 'full', 'fullslow', 'unfreeze'
]


def get_model_size(model):
    if model in SIZE_MAP:
        model_size = SIZE_MAP[model]
    else:
        model_size = model.split('.')[-1].split('_')[0]
    if model_size not in SRC_MODELS:
        print(f'unknown model size "{model_size}" in model "{model}"')
        exit(1)
    return model_size


def sort_models(names):
    new_names = []

    # Group by size
    by_size = {s: [] for s in MODEL_SIZES}
    for n in names:
        s = get_model_size(n)
        by_size[s].append(n)

    for s in MODEL_SIZES:
        names = by_size[s]

        # Group by method
        by_method = {m: [] for m in METHODS}
        for n in names:
            m = re.sub(r'\d+$', '', n.split('_')[-1])
            if m in MODEL_SIZES:
                m = 'orig'
            if m not in METHODS:
                print(f'WARNING: skipping unknown method "{m}"')
                continue
            by_method[m].append(n)

        for m in METHODS:
            names = by_method[m]
            names = sorted(names)

            new_names.extend(names)
    return new_names
