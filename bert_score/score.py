import os
import time
import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

from .utils import (get_idf_dict, bert_cos_score_idf,
                    get_bert_embedding, model_types,
                    lang2model, model2layers, get_hash)


__all__ = ['score', 'plot_example']

def get_model(model_type, num_layers, all_layers=None):
    model = AutoModel.from_pretrained(model_type)
    model.eval()

    # drop unused layers
    if not all_layers:
        if 'bert' == model_type[:4] or 'roberta' == model_type[:7]:
            model.encoder.layer =\
                torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif 'xlnet' == model_type[:5]:
            model.layer =\
                torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif 'xlm' in model_type[:3]:
            model.n_layers = num_layers
        else:
            raise ValueError("Not supported")
    else:
        if 'bert' == model_type[:4] or 'roberta' == model_type[:7]:
            model.encoder.output_hidden_states = True
        else:
            model.output_hidden_states = True
    return model

def score(cands, refs, model_type=None, num_layers=None, verbose=False,
          idf=False, batch_size=64, nthreads=4, all_layers=False, lang=None,
          return_hash=False):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str): reference sentences
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool): use idf weighting
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences
        - :param: `return_hash` (bool): return hash code of the setting
    """
    assert len(cands) == len(refs)

    assert lang is not None or model_type is not None, \
        'Either lang or model_type should be specified'

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]


    assert model_type in model_types
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = get_model(model_type, num_layers, all_layers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    else:
        if verbose:
            print('preparing IDF dict...')
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print('done in {:.2f} seconds'.format(time.perf_counter() - start))

    if verbose:
        print('calculating scores...')
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict,
                                   verbose=verbose, device=device, 
                                   batch_size=batch_size, all_layers=all_layers)

    P = all_preds[..., 0].cpu()
    R = all_preds[..., 1].cpu()
    F1 = all_preds[..., 2].cpu()
    if verbose:
        time_diff = time.perf_counter() - start
        print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

    if return_hash:
        return (P, R, F1), get_hash(model_type, num_layers, idf)
    else:
        return P, R, F1

# Under Construction
def plot_example(candidate, reference, model_type=None, lang=None, num_layers=None, fname=''):
    """
    BERTScore metric.

    Args:
        - :param: `h` (str): a candidate sentence
        - :param: `r` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
    """
    assert isinstance(candidate, str)
    assert isinstance(reference, str)

    assert lang is not None or model_type is not None, \
        'Either lang or model_type should be specified'

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    assert model_type in model_types
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = get_model(model_type, num_layers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    idf_dict = defaultdict(lambda: 1.)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    hyp_embedding, masks, padded_idf = get_bert_embedding([candidate], model, tokenizer, idf_dict,
                                                         device=device, all_layers=False)
    ref_embedding, masks, padded_idf = get_bert_embedding([reference], model, tokenizer, idf_dict,
                                                         device=device, all_layers=False)
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim.squeeze(0).cpu()

    # remove [CLS] and [SEP] tokens 
    r_tokens = [tokenizer.decode([i]) for i in tokenizer.encode(reference)]
    h_tokens = [tokenizer.decode([i]) for i in tokenizer.encode(candidate)]
    sim = sim[1:-1,1:-1]

    fig, ax = plt.subplots(figsize=(len(r_tokens)*0.8, len(h_tokens)*0.8))
    im = ax.imshow(sim, cmap='Blues')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(r_tokens)))
    ax.set_yticks(np.arange(len(h_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(r_tokens, fontsize=10)
    ax.set_yticklabels(h_tokens, fontsize=10)
    plt.xlabel("Refernce (tokenized)", fontsize=14)
    plt.ylabel("Candidate (tokenized)", fontsize=14)
    plt.title("Similarity Matrix", fontsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_tokens)):
        for j in range(len(r_tokens)):
            text = ax.text(j, i, '{:.3f}'.format(sim[i, j].item()),
                           ha="center", va="center", color="k" if sim[i, j].item() < 0.6 else "w")

    fig.tight_layout()
    if fname != "":
        print("Saved figure to file: ", fname)
        plt.savefig(fname, dpi=100)
    plt.show()
