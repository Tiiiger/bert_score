#!/usr/bin/env python
import os
import time
import argparse
import torch
from collections import defaultdict

import bert_score


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Visualize BERTScore')
    parser.add_argument('--lang', type=str, default='en', help='two-letter abbreviation of the language (e.g., en)')
    parser.add_argument('-m', '--model', default=None, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', type=int, default=None, help='use first N layer in BERT (default: 8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('-r', '--ref', type=str, required=True, help='reference sentence')
    parser.add_argument('-c', '--cand', type=str, required=True, help='candidate sentence')
    parser.add_argument('-f', '--file', type=str, default='visualize.png', help='name of file to save output matrix in')
    parser.add_argument('--rescale-with-baseline', action='store_true', help='Rescaling the numerical score with precomputed baselines')

    args = parser.parse_args()
 
    bert_score.plot_example(args.cand, args.ref, model_type=args.model, 
        lang=args.lang, num_layers=args.num_layers, fname=args.file,
        rescale_with_baseline=args.rescale_with_baseline)


if __name__ == "__main__":
    main()
