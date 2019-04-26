#!/usr/bin/env python
import os
import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import bert_score

VERSION=bert_score.__version__

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate BERTScore')
    parser.add_argument('--bert', default='bert-base-multilingual-cased',
                        choices=bert_score.bert_types, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', default=8, help='use first N layer in BERT (default: 8)')
    parser.add_argument('-b', '--batch_size', default=64, help='batch size (default: 64)')
    parser.add_argument('--no_idf', action='store_true', help='BERT Score without IDF scaling')
    parser.add_argument('-s', '--seg_level', action='store_true', help='show individual score of each pair')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('-r', '--ref', required=True, help='reference file path or a string')
    parser.add_argument('-c', '--cand', required=True,help='candidate (system outputs) file path or a string')

    args = parser.parse_args()

    if os.path.isfile(args.cand) and os.path.isfile(args.ref):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        with open(args.ref) as f:
            refs = [line.strip() for line in f]
    else:
        cands = [args.cand]
        refs = [args.ref]
        assert args.no_idf, "do not suuport idf fold for a single pair of sentences"

    assert len(cands) == len(refs)

    all_preds = bert_score.score(cands, refs, bert=args.bert, num_layers=args.num_layers, verbose=args.verbose,
                                 no_idf=args.no_idf, batch_size=args.batch_size)
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    if args.verbose:
        print('done in {:.2f} seconds'.format(time.perf_counter() - start))
    msg = '{}_L{}{}_version={} BERT-P: {:.6f} BERT-R: {:.6f} BERT-F1: {:.6f}'.format(
        args.bert, args.num_layers, '_no-idf' if args.no_idf else '', VERSION, P, R, F1)
    print(msg)
    if args.seg_level:
        for p, r, f in all_preds.tolist():
            print('{:.6f}\t{:.6f}\t{:.6f}'.format(p, r, f))


if __name__ == "__main__":
    main()
