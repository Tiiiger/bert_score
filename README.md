# BERTScore
Automatic Evaluation Metric described in the paper [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675).

#### Authors:
* [Tianyi Zhang](https://scholar.google.com/citations?user=OI0HSa0AAAAJ&hl=en)*
* Varsha Kishore
* [Felix Wu](https://scholar.google.com.tw/citations?user=sNL8SSoAAAAJ&hl=en)*
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
Install requiremnts by `pip install -r requiremnts.txt`

Install it from the source by:
```sh
git clone https://github.com/Tiiiger/bert_score
cd bert_score
pip install .
```

### Usage

#### Metric
We provide a command line interface(CLI) of BERTScore as well as a python module. 
For the CLI, you can use it as follows:
1. To evaluate English text files:

We provide example inputs under `./example`.

```sh
bert-score -r example/refs.txt -c example/hyps.txt --bert bert-base-uncased 
```
2. To evaluate Chinese text files:

Please format your input files similar to the ones in `./example`.

```sh
bert-score -r [references] -c [candidates] --bert bert-base-chinese
```
3. To evaluate text files in other languages:

Please format your input files similar to the ones in `./example`.

```sh
bert-score -r [references] -c [candidates]
```
See more options by `bert-score -h`.

For the python module, please refer to [`cli/score.py`](https://github.com/Tiiiger/bert_score/blob/master/cli/score.py).

<!---
#### Visualization
Because BERTScore measure sentence similarity by accumulating word similarites,
we can visualize it easily.
--->
<!---
Below is an example where we visualize the pairwise cosine similarity of words
in the reference and candidate sentences.
--->
<!-- ![]() -->

### Acknowledgement
This repo wouldn't be possible without the awesome [bert](https://github.com/google-research/bert) and [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
