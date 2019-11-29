# BERTScore
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![Downloads](https://pepy.tech/badge/bert-score)](https://pepy.tech/project/bert-score) [![Downloads](https://pepy.tech/badge/bert-score/month)](https://pepy.tech/project/bert-score/month) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic Evaluation Metric described in the paper [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675).
#### News:
- Updated to version 0.2.2
  - **Bug fixed**: when using RoBERTaTokenizer, we now set `add_prefix_space=True` which was the default setting in huggingface's `pytorch_transformers` (when we ran the experiments in the paper) before they migrated it to `transformers`. This breaking change in `transformers` leads to a lower correlation with human evalutation. To reproduce our RoBERTa results in the paper, please use version `0.2.2`.
  - the best number of layers for DistilRoBERTa is included
  - support loading a custom model
- Updated to version 0.2.1
  - [SciBERT](https://github.com/allenai/scibert) (Beltagy et al.) models are now included. Thanks to AI2 for sharing the models. By default, we use the 9th layer (the same as BERT-base), but this is not tuned. 
- Our [arXiv paper](https://arxiv.org/abs/1904.09675) has been updated to v2 with more experiments and analysis.
- Updated to version 0.2.0
  - Supporting BERT, XLM, XLNet, and RoBERTa models using [huggingface's Transformers library](https://github.com/huggingface/transformers)
  - Automatically picking the best model for a given language
  - Automatically picking the layer based a model
  - IDF is *not* set as default as we show in the new version that the improvement brought by importance weighting is not consistent

#### Authors:
* [Tianyi Zhang](https://scholar.google.com/citations?user=OI0HSa0AAAAJ&hl=en)*
* Varsha Kishore*
* [Felix Wu](https://sites.google.com/view/felixwu/home)*
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)
* [Yoav Artzi](https://yoavartzi.com/)

*: Equal Contribution

### Overview
BERTScore leverages the pre-trained contextual embeddings from BERT and matches
words in candidate and reference sentences by cosine similarity.
It has been shown to correlate with human judgment on setence-level and
system-level evaluation.
Moreover, BERTScore computes precision, recall, and F1 measure, which can be
useful for evaluating different language generation tasks.

For an illustration, BERTScore precision can be computed as
![](./bert_score.png "BERTScore")

If you find this repo useful, please cite:
```
@article{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Zhang, Tianyi and Kishore, Varsha and Wu, Felix and Weinberger, Kilian Q. and Artzi, Yoav.},
  journal={arXiv preprint arXiv:1904.09675},
  year={2019}
}
```

### Installation
* Python version >= 3.6
* PyTorch version >= 1.0.0

Install from pip by 

```sh
pip install bert-score
```

Install it from the source by:
```sh
git clone https://github.com/Tiiiger/bert_score
cd bert_score
pip install .
```
and you may test your installation by:
```
python -m unittest discover
```

### Usage

#### Command Line Interface (CLI)
We provide a command line interface (CLI) of BERTScore as well as a python module. 
For the CLI, you can use it as follows:
1. To evaluate English text files:

We provide example inputs under `./example`.

```sh
bert-score -r example/refs.txt -c example/hyps.txt --lang en
```
You will get the following output at the end:

roberta-large_L17_no-idf_version=0.2.2 BERT-P: 0.950530 BERT-R: 0.949223 BERT-F1: 0.949839

where "roberta-large_L17_no-idf_version=0.2.2" is the hashcode.

2. To evaluate text files in other languages:

We currently support the 104 languages in multilingual BERT ([full list](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)).

Please specify the two-letter abbrevation of the language. For instance, using `--lang zh` for Chinese text. 

See more options by `bert-score -h`.

3. To load your own custom model:
Please specify the path to the model and the number of layers to use by `--model` and `--num_layers`.
```sh
bert-score -r example/refs.txt -c example/hyps.txt --model path_to_my_bert --num_layers 9
```

4. To visualize matching scores:
```sh
bert-score-show --lang en -r "There are two bananas on the table." -c "On the table are two apples." -f out.png
```
The figure will be saved to out.png.

#### Python Function
For the python module, we provide a [demo](./example/Demo.ipynb). 
Please refer to [`bert_score/score.py`](./bert_score/score.py) for more details.

Running BERTScore can be computationally intensive (because it uses BERT :p).
Therefore, a GPU is usually necessary. If you don't have access to a GPU, you
can try our [demo on Google Colab](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q)


#### Practical Tips

* Report the hash code (e.g., `roberta-large_L17_no-idf_version=0.2.1`) in your paper so that people know what setting you use. This is inspired by [sacreBLEU](https://github.com/mjpost/sacreBLEU).
* Unlike BERT, RoBERTa uses GPT2-style tokenizer which creates addition " " tokens when there are multiple spaces appearing together. It is recommended to remove addition spaces by `sent = re.sub(r' +', ' ', sent)` or `sent = re.sub(r'\s+', ' ', sent)`.
* Using inverse document frequency (idf) on the reference
  sentences to weigh word importance  may correlate better with human judgment.
  However, when the set of reference sentences become too small, the idf score 
  would become inaccurate/invalid.
  We now make it optional. To use idf,
  please set `--idf` when using the CLI tool or
  `idf=True` when calling `bert_score.score` function.
* When you are low on GPU memory, consider setting `batch_size` when calling
  `bert_score.score` function.
* To use a particular model please set `-m MODEL_TYPE` when using the CLI tool
  or `model_type=MODEL_TYPE` when calling `bert_score.score` function. 
* We tune layer to use based on WMT16 metric evaluation dataset. You may use a
  different layer by setting `-l LAYER` or `num_layers=LAYER`
* __Limitation__: Because BERT, RoBERTa, and XLM with learned positional embeddings are pre-trained on sentences with max length 512, BERTScore is undefined between sentences longer than 510 (512 after adding \[CLS\] and \[SEP\] tokens). The sentences longer than this will be truncated. Please consider using XLNet which can support much longer inputs.

### Default Behavior

#### Default Model
| Language  | Model                        |
|:---------:|:----------------------------:|
| en        | roberta-large                |
| en-sci    | scibert-scivocab-uncased     |
| zh        | bert-base-chinese            |
| others    | bert-base-multilingual-cased |

#### Default Layers
| Model                           | Best Layer | Max Length    |
|:-------------------------------:|------------| ------------- |
| bert-base-uncased               | 9          | 512           |
| bert-large-uncased              | 18         | 512           |
| bert-base-cased-finetuned-mrpc  | 9          | 512           |
| bert-base-multilingual-cased    | 9          | 512           |
| bert-base-chinese               | 8          | 512           |
| roberta-base                    | 10         | 512           |
| roberta-large                   | 17         | 512           |
| roberta-large-mnli              | 19         | 512           |
| xlnet-base-cased                | 5          | 1000000000000 |
| xlnet-large-cased               | 7          | 1000000000000 |
| xlm-mlm-en-2048                 | 7          | 512           |
| xlm-mlm-100-1280                | 11         | 512           |
| scibert-scivocab-uncased        | 9*         | 512           |
| scibert-scivocab-cased          | 9*         | 512           |
| scibert-basevocab-uncased       | 9*         | 512           |
| scibert-basevocab-cased         | 9*         | 512           |
| distilroberta-base              | 5          | 512           |

*: Not tuned

### Acknowledgement
This repo wouldn't be possible without the awesome
[bert](https://github.com/google-research/bert), [fairseq](https://github.com/pytorch/fairseq), and [transformers](https://github.com/huggingface/transformers).
