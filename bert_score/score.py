import os
import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .utils import get_idf_dict, bert_cos_score_idf,\
                   get_bert_embedding, bert_types

__all__ = ['score', 'plot_example']

def score(cands, refs, bert="bert-base-multilingual-cased",
          num_layers=8, verbose=False, no_idf=False, batch_size=64):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str): reference sentences
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `no_idf` (bool): do not use idf weighting
        - :param: `batch_size` (int): bert score processing batch size
    """
    assert len(cands) == len(refs)
    assert bert in bert_types

    tokenizer = BertTokenizer.from_pretrained(bert)
    model = BertModel.from_pretrained(bert)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # drop unused layers
    model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

    if no_idf:
        idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[101] = 0
        idf_dict[102] = 0
    else:
        if verbose:
            print('preparing IDF dict...')
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer)
        if verbose:
            print('done in {:.2f} seconds'.format(time.perf_counter() - start))

    if verbose:
        print('calculating scores...')
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict,
                                   verbose=verbose, device=device, batch_size=batch_size)

    P = all_preds[:, 0].cpu()
    R = all_preds[:, 1].cpu()
    F1 = all_preds[:, 2].cpu()
    if verbose:
        print('done in {:.2f} seconds'.format(time.perf_counter() - start))

    return P, R, F1

def plot_example(h, r, verbose=False, bert="bert-base-multilingual-cased",
                 num_layers=8, fname=''):
    """
    BERTScore metric.

    Args:
        - :param: `h` (str): a candidate sentence
        - :param: `r` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
    """
    assert bert in bert_types

    if verbose:
        print('loading BERT model...')
    tokenizer = BertTokenizer.from_pretrained(bert)
    model = BertModel.from_pretrained(bert)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    h_tokens = ['[CLS]'] + tokenizer.tokenize(h) + ['[SEP]']
    r_tokens = ['[CLS]'] + tokenizer.tokenize(r) + ['[SEP]']

    model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
    idf_dict = defaultdict(lambda: 1.)

    ref_embedding, ref_lens, ref_masks, padded_idf = get_bert_embedding([r], model, tokenizer, idf_dict,
                                       device=device)
    hyp_embedding, ref_lens, ref_masks, padded_idf = get_bert_embedding([h], model, tokenizer, idf_dict,
                                       device=device)

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(1)

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2)).cpu()
    sim = sim.squeeze(0).numpy()

    # remove [CLS] and [SEP] tokens 
    r_tokens = r_tokens[1:-1]
    h_tokens = h_tokens[1:-1]
    sim = sim[1:-1,1:-1]

    fig, ax = plt.subplots(figsize=(len(r_tokens)*0.8, len(h_tokens)*0.8))
    im = ax.imshow(sim, cmap='Blues')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(r_tokens)))
    ax.set_yticks(np.arange(len(h_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(r_tokens, fontsize=10)
    ax.set_yticklabels(h_tokens, fontsize=10)
    plt.xlabel("Refernce", fontsize=10)
    plt.ylabel("Candidate", fontsize=10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_tokens)):
        for j in range(len(r_tokens)):
            text = ax.text(j, i, '{:.3f}'.format(sim[i, j]),
                           ha="center", va="center", color="k" if sim[i, j] < 0.6 else "w")

#     P = sim.max(1).mean()
#     R = sim.max(0).mean()
#     F1 = 2 * P * R / (P + R)

    fig.tight_layout()
#     plt.title("BERT-F1: {:.3f}".format(F1), fontsize=10)
    if fname != "":
        print("Saved figure to file: ", fname+".png")
        plt.savefig(fname+'.png', dpi=100)
    plt.show()
