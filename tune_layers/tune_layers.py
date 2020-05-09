import os
import re
import argparse
import torch
import numpy as np

from tqdm.auto import tqdm
from collections import defaultdict
from scipy.stats import pearsonr

import bert_score


def get_wmt16(lang_pair, data_folder="wmt16"):
    with open(
        os.path.join(
            data_folder,
            f"wmt16-metrics-results/seg-level-results/DAseg-newstest2016/DAseg-newstest2016.human.{lang_pair}",
        )
    ) as f:
        gold_scores = list(map(float, f.read().strip().split("\n")))

    with open(
        os.path.join(
            data_folder,
            f"wmt16-metrics-results/seg-level-results/DAseg-newstest2016/DAseg-newstest2016.reference.{lang_pair}",
        )
    ) as f:
        all_refs = f.read().strip().split("\n")

    with open(
        os.path.join(
            data_folder,
            f"wmt16-metrics-results/seg-level-results/DAseg-newstest2016/DAseg-newstest2016.mt-system.{lang_pair}",
        )
    ) as f:
        all_hyps = f.read().strip().split("\n")

    return gold_scores, all_refs, all_hyps


def get_wmt16_seg_to_bert_score(lang_pair, network, num_layers, idf=False, cache=False, data_folder="wmt16"):
    os.makedirs(f"cache_score/{network}", exist_ok=True)
    path = "cache_score/{}/wmt16_seg_to_{}_{}.pkl".format(network, *lang_pair.split("-"))

    gold_scores, refs, cands = get_wmt16(lang_pair, data_folder=data_folder)
    model_type = network
    scores_idf = bert_score.score(
        cands, refs, model_type=model_type, num_layers=num_layers, verbose=False, idf=idf, all_layers=True
    )
    scores = list(scores_idf)

    return scores, gold_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="wmt16", help="path to wmt16 data")
    parser.add_argument("-m", "--model", nargs="+", help="models to tune")
    parser.add_argument("-l", "--log_file", default="best_layers_log.txt", help="log file path")
    parser.add_argument("--idf", action="store_true")
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        default=["cs-en", "de-en", "fi-en", "ro-en", "ru-en", "tr-en"],
        help="language pairs used for tuning",
    )
    args = parser.parse_args()

    networks = args.model
    for network in networks:
        results = defaultdict(dict)
        for lang_pair in tqdm(args.lang_pairs):
            scores, gold_scores = get_wmt16_seg_to_bert_score(lang_pair, network, 100, idf=args.idf, cache=False)
            for i, score in enumerate(scores[2]):
                results[lang_pair + " " + str(i)]["%s %s" % (network, "F")] = pearsonr(score, gold_scores)[0]

        best_layer, best_corr = 0, 0.0
        for num_layer in range(100):
            temp = []
            if f"{args.lang_pairs[0]} {num_layer}" not in results:
                break
            for lp in args.lang_pairs:
                temp.append(results[f"{lp} {num_layer}"][f"{network} F"])
            corr = np.mean(temp)
            results["avg" + " " + str(num_layer)]["%s %s" % (network, "F")] = corr
            if corr > best_corr:
                best_layer, best_corr = num_layer, corr
        if args.idf:
            msg = f"'{network}' (idf): {best_layer}, # {best_corr}"
        else:
            msg = f"'{network}': {best_layer}, # {best_corr}"
        print(msg)
        with open(args.log_file, "a") as f:
            print(msg, file=f)


if __name__ == "__main__":
    main()
