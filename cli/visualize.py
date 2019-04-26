#!/usr/bin/env python
import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import bert_score

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate BERTScore')
    parser.add_argument('--bert', default='bert-base-multilingual-cased',
                        choices=bert_score.bert_types, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', default=8, help='use first N layer in BERT (default: 8)')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('-r', '--ref', required=True, help='reference sentence')
    parser.add_argument('-c', '--cand', required=True,help='candidate sentence')
    parser.add_argument('-o', '--output_file_name', default='',help='output file name')

    args = parser.parse_args()

    cand = args.cand
    ref = args.ref
    fname = args.output_file_name
    bert_score.plot_example(cand, ref, verbose=args.verbose,
                            bert=args.bert, num_layers=args.num_layers,
                            fname = fname)


if __name__ == "__main__":
    main()
