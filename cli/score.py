#!/usr/bin/env python
import os
import time
import argparse
import torch
from collections import defaultdict

import bert_score


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate BERTScore')
    parser.add_argument('--lang', type=str, default=None, help='two-letter abbreviation of the language (e.g., en)')
    parser.add_argument('-m', '--model', default=None,
                        choices=bert_score.model_types, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', type=int, default=None, help='use first N layer in BERT (default: 8)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--idf', action='store_true', help='BERT Score with IDF scaling')
    parser.add_argument('-s', '--seg_level', action='store_true', help='show individual score of each pair')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('-r', '--ref', type=str, required=True, help='reference file path or a string')
    parser.add_argument('-c', '--cand', type=str, required=True, help='candidate (system outputs) file path or a string')

    args = parser.parse_args()

    if os.path.isfile(args.cand) and os.path.isfile(args.ref):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        with open(args.ref) as f:
            refs = [line.strip() for line in f]
    else:
        cands = [args.cand]
        refs = [args.ref]
        assert not args.idf, "do not suuport idf fold for a single pair of sentences"

    assert len(cands) == len(refs)

    all_preds = bert_score.score(cands, refs, model_type=args.model, num_layers=args.num_layers,
                                            verbose=args.verbose, idf=args.idf, batch_size=args.batch_size,
                                            lang=args.lang, return_hash=True)
    all_preds, hash_code = all_preds[:3], all_preds[3]
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    msg = hash_code + \
        f' BERT-P: {P:.6f} BERT-R: {R:.6f} BERT-F1: {F1:.6f}'
    print(msg)
    if args.seg_level:
        ps, rs, fs = all_preds
        for p, r, f in zip(ps, rs, fs):
            print('{:.6f}\t{:.6f}\t{:.6f}'.format(p, r, f))


if __name__ == "__main__":
    main()
