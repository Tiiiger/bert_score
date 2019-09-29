# BERTScore
Automatic Evaluation Metric described in the paper [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675).
#### News:
- Updated to version 0.2.0
  - Supporting BERT, XLM, XLNet, and RoBERTa models using [huggingface's Transformers library](https://github.com/huggingface/transformers)
  - Automatically picking the best model for a given language
  - Automatically picking the layer based a model
  - IDF is *not* set as default 
  - *plot_example* is under construction  

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
![](https://github.com/Tiiiger/bert_score/blob/master/bert_score.png "BERTScore")

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

roberta-large_L17_no-idf_version=0.2.0 BERT-P: 0.950530 BERT-R: 0.949223 BERT-F1: 0.949839

2. To evaluate text files in other languages:

We currently support the 104 languages in multilingual BERT ([full list](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)).

Please specify the two-letter abbrevation of the language. For instance, using `--lang zh` for Chinese text. 

See more options by `bert-score -h`.

#### Python Function
For the python module, we provide a [demo](./example/Demo.ipynb). 
Please refer to [`bert_score/score.py`](./bert_score/score.py) for more details.

Running BERTScore can be computationally intensive (because it uses BERT :p).
Therefore, a GPU is usually necessary. If you don't have access to a GPU, you
can try our [demo on Google Colab](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q)


#### Practical Tips

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

### Default Behavior

#### Default Model
| Language  | Model                        |
|:---------:|:----------------------------:|
| en        | roberta-large                |
| zh        | bert-base-chinese            |
| others    | bert-base-multilingual-cased | 

#### Default Layers
| Model                           | Best Layer |
|:-------------------------------:|------------|
| bert-base-uncased               | 9          |
| bert-large-uncased              | 18         |
| bert-base-cased-finetuned-mrpc  | 9          |
| bert-base-multilingual-cased    | 9          |
| bert-base-chinese               | 8          |
| roberta-base                    | 10         |
| roberta-large                   | 17         |
| roberta-large-mnli              | 19         |
| xlnet-base-cased                | 5          | 
| xlnet-large-cased               | 7          | 
| xlm-mlm-en-2048                 | 7          | 
| xlm-mlm-100-1280                | 11         |

### Acknowledgement
This repo wouldn't be possible without the awesome
[bert](https://github.com/google-research/bert) and
[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
