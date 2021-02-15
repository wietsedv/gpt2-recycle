# GPT-2 Recycled for Italian and Dutch
[Wietse de Vries](https://www.semanticscholar.org/author/Wietse-de-Vries/144611157) •
[Malvina Nissim](https://www.semanticscholar.org/author/M.-Nissim/2742475)

## Model description

In our paper, we describe a multi-stage adaptation method for transfering GPT-2 to Italian and Dutch without unnecessary retraining. This repository contains the source code and the final models are available on the Hugging Face model hub (see below).

We publish two types of models:
 - Models where only the lexical layer is retrained for the new language and the Transformer layers are the same as the English model. The lexical layers of these models are in practice automatically aligned with the equivalent English model. Use this if you are interested in alignment properties.
 - Models with retrained lexical embeddings and then additional training of the full models. Use this if you want to generate more realistic text.

For details, check out our paper on [arXiv](https://arxiv.org/abs/2012.05628) and the models on the [🤗 Hugging Face model hub](https://huggingface.co/GroNLP) (see links for specific models below).


## Models

### Dutch
 - [`gpt2-small-dutch-embeddings`](https://huggingface.co/GroNLP/gpt2-small-dutch-embeddings): Small model size with only retrained lexical embeddings.
 - [`gpt2-small-dutch`](https://huggingface.co/GroNLP/gpt2-small-dutch):  Small model size with retrained lexical embeddings and additional fine-tuning of the full model. (**Recommended**)
 - [`gpt2-medium-dutch-embeddings`](https://huggingface.co/GroNLP/gpt2-medium-dutch-embeddings): Medium model size with only retrained lexical embeddings.

### Italian
 - [`gpt2-small-italian-embeddings`](https://huggingface.co/GroNLP/gpt2-small-italian-embeddings): Small model size with only retrained lexical embeddings.
 - [`gpt2-small-italian`](https://huggingface.co/GroNLP/gpt2-small-italian):  Small model size with retrained lexical embeddings and additional fine-tuning of the full model. (**Recommended**)
 - [`gpt2-medium-italian-embeddings`](https://huggingface.co/GroNLP/gpt2-medium-italian-embeddings): Medium model size with only retrained lexical embeddings.


## How to use

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="GroNLP/gpt2-small-dutch")
print(pipe('Was ik maar een'))
```

```python
from transformers import AutoTokenizer, AutoModel, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch")
model = AutoModel.from_pretrained("GroNLP/gpt2-small-dutch")  # PyTorch
model = TFAutoModel.from_pretrained("GroNLP/gpt2-small-dutch")  # Tensorflow
```

## BibTeX entry

```bibtex
@misc{devries2020good,
      title={As good as new. How to successfully recycle English GPT-2 to make models for other languages}, 
      author={Wietse de Vries and Malvina Nissim},
      year={2020},
      eprint={2012.05628},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
