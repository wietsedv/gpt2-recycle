# How to recycle GPT-2 with the code in this repository

Here are some guidelines for using code in this repository for recycling GPT-2 to a target language of your choice. Note that libraries like HuggingFace Transformers have improved a lot since the initial start of this project and most of the steps can be reproduced more simply with current version of the Transformers and Datasets libraries of Huggingface.

## Tokenization and data preparation

Data preparation depends on your data and use case. As a guideline you can follow the scripts in `src/preparation` in order. These scripts do in order:

 1. Create a BPE vocabulary.
 2. Sanity checks for single characters, sentence lengths and token coverage in the training data.
 3. Tokenize the full data.
 4. Split into training and validation subsets. (Test should not be sampled from training data.)

## Transform original embeddings to new vocabulary

Instead of randomly initializing a new lexical layer for the new language, you can transform the existing English lexical layer to your target language. This can be done with `python src/alignment/transform.py --help`. Check out the paper for differences between transformation methods.

Other scripts for alignment based evaluation (like dictionary induction) are in `src/alignment`.

## Train the lexical layer (and the full model)

The original code used for lexical layer retraining is available in `src/training/main.py`. Run `python src/training/main.py --help` to see all training options.

Initially, you only want to tune the lexical (wte) layer. Do this with the `--wte_only` argument of the training script. After loss has stopped decreasing, you can further train the checkpoint without that option (and a low learning rate) to adapt the Transformer layers to your language.

## Evaluate on out-of-domain data
The `src/evaluation` directory contains some starting points for evaluation. This includes (strided) perplexity calculation on test data (`ppl.py`) and example generation `generate.py`.
