import os
import pathlib
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer

from .utils import (bert_cos_score_idf, cache_scibert, get_bert_embedding,
                    get_hash, get_idf_dict, get_model, get_tokenizer,
                    lang2model, model2layers, sent_encode)
from .scorer import BERTScorer

__all__ = ["score", "plot_example"]


def score(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    dtype: str = "fp32",
    grad_enabled=False,
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `dtype` (str): dtype 'fp32', 'fp16', 'bf16', or 'int8'
        - :param: `grad_enbabled` (bool): enable gradients

    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    if idf:
        idf_sents = refs if isinstance(refs[0], str) else [s for ref in refs for s in ref]
    else:
        idf_sents = None
    scorer = BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        nthreads=nthreads,
        all_layers=all_layers,
        idf=idf,
        idf_sents=idf_sents,
        device=device,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
        use_fast_tokenizer=use_fast_tokenizer,
        dtype=dtype,
    )

    return scorer.score(
        cands=cands,
        refs=refs,
        verbose=verbose,
        batch_size=batch_size,
        return_hash=return_hash,
        grad_enabled=grad_enabled,
    )


def plot_example(
    candidate,
    reference,
    model_type=None,
    num_layers=None,
    lang=None,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    fname="",
):    
    """
    BERTScore plot.

    Args:
        - :param: `candidate` (str): a candidate sentence
        - :param: `reference` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `fname` (str): path to save the output plot
    """
    scorer = BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
        use_fast_tokenizer=use_fast_tokenizer,
    )

    return scorer.plot_example(
        candidate=candidate,
        reference=reference,
        fname=fname,
    )
