import os
import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from .utils import get_idf_dict, bert_cos_score_idf

__all__ = ['score']

def score(cands, refs, bert="bert-base-multilingual-cased", num_layers=8, verbose=False, no_idf=False, batch_size=64):
    assert len(cands) == len(refs)

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
    all_preds = bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict, device=device,
                                   batch_size=batch_size)

    P = all_preds[:, 0].cpu()
    R = all_preds[:, 1].cpu()
    F1 = all_preds[:, 2].cpu()
    if verbose:
        print('done in {:.2f} seconds'.format(time.perf_counter() - start))

    return P, R, F1
