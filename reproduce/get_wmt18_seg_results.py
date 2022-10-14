import argparse
import os
import pickle as pkl
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm.auto import tqdm, trange

import bert_score

wmt18_sys_to_lang_pairs = [
    "cs-en",
    "de-en",
    "et-en",
    "fi-en",
    "ru-en",
    "tr-en",
    "zh-en",
]
wmt18_sys_from_lang_pairs = [
    "en-cs",
    "en-de",
    "en-et",
    "en-fi",
    "en-ru",
    "en-tr",
    "en-zh",
]
wmt18_sys_lang_pairs = wmt18_sys_to_lang_pairs + wmt18_sys_from_lang_pairs

import argparse


def get_wmt18_seg_data(lang_pair):
    src, tgt = lang_pair.split("-")

    RRdata = pd.read_csv(
        "wmt18/wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv", sep=" "
    )
    # RRdata_lang = RRdata[RRdata['LP'] == lang_pair] # there is a typo in this data. One column name is missing in the header
    RRdata_lang = RRdata[RRdata.index == lang_pair]

    systems = set(RRdata_lang["BETTER"])
    systems.update(list(set(RRdata_lang["WORSE"])))
    systems = list(systems)
    sentences = {}
    for system in systems:
        with open(
            "wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/system-outputs/newstest2018/{}/newstest2018.{}.{}".format(
                lang_pair, system, lang_pair
            )
        ) as f:
            sentences[system] = f.read().split("\n")

    with open(
        "wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/"
        "references/{}".format("newstest2018-{}{}-ref.{}".format(src, tgt, tgt))
    ) as f:
        references = f.read().split("\n")

    ref = []
    cand_better = []
    cand_worse = []
    for index, row in RRdata_lang.iterrows():
        cand_better += [sentences[row["BETTER"]][row["SID"] - 1]]
        cand_worse += [sentences[row["WORSE"]][row["SID"] - 1]]
        ref += [references[row["SID"] - 1]]

    return ref, cand_better, cand_worse


def kendell_score(scores_better, scores_worse):
    total = len(scores_better)
    correct = torch.sum(scores_better > scores_worse).item()
    incorrect = total - correct
    return (correct - incorrect) / total


def get_wmt18_seg_bert_score(
    lang_pair, scorer, cache=False, from_en=True, batch_size=64
):
    filename = ""
    if from_en:
        if scorer.idf:
            filename = "cache_score/from_en/18/{}/wmt18_seg_from_{}_{}_idf.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
        else:
            filename = "cache_score/from_en/18/{}/wmt18_seg_from_{}_{}.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
    else:
        if scorer.idf:
            filename = "cache_score/to_en/18/{}/wmt18_seg_to_{}_{}_idf.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
        else:
            filename = "cache_score/to_en/18/{}/wmt18_seg_to_{}_{}.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pkl.load(f)
    else:
        refs, cand_better, cand_worse = get_wmt18_seg_data(lang_pair)
        if scorer.idf:
            scorer.compute_idf(refs)
        scores_better = list(scorer.score(cand_better, refs, batch_size=batch_size))
        scores_worse = list(scorer.score(cand_worse, refs, batch_size=batch_size))
        if cache:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pkl.dump((scores_better, scores_worse), f)
        return scores_better, scores_worse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="wmt18", help="path to wmt16 data")
    parser.add_argument("-m", "--model", nargs="+", help="models to tune")
    parser.add_argument(
        "-l", "--log_file", default="wmt18_log.csv", help="log file path"
    )
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        default=wmt18_sys_to_lang_pairs,
        help="language pairs used for tuning",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    header = "model_type"
    for lang_pair in args.lang_pairs + ["avg"]:
        header += f",{lang_pair}"
    print(header)
    if not os.path.exists(args.log_file):
        with open(args.log_file, "w") as f:
            print(header, file=f)

    print(args.model)
    for model_type in args.model:
        scorer = bert_score.scorer.BERTScorer(model_type=model_type, idf=args.idf)
        results = defaultdict(dict)
        for lang_pair in tqdm(args.lang_pairs):
            scores_better, scores_worse = get_wmt18_seg_bert_score(
                lang_pair, scorer, batch_size=args.batch_size, cache=True, from_en=False
            )
            for sb, sw, name in zip(scores_better, scores_worse, ["P", "R", "F"]):
                results[lang_pair][f"{model_type} {name}"] = kendell_score(sb, sw)

        for name in ["P", "R", "F"]:
            temp = []
            for lang_pair in args.lang_pairs:
                temp.append(results[lang_pair][f"{model_type} {name}"])
            results["avg"][f"{model_type} {name}"] = np.mean(temp)

            msg = f"{model_type} {name} (idf)" if args.idf else f"{model_type} {name}"
            for lang_pair in args.lang_pairs + ["avg"]:
                msg += f",{results[lang_pair][f'{model_type} {name}']}"
            print(msg)
            with open(args.log_file, "a") as f:
                print(msg, file=f)

        del scorer


if __name__ == "__main__":
    main()
