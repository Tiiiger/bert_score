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

wmt17_sys_to_lang_pairs = [
    "cs-en",
    "de-en",
    "fi-en",
    "lv-en",
    "ru-en",
    "tr-en",
    "zh-en",
]
wmt17_sys_from_lang_pairs = ["en-cs", "en-de", "en-lv", "en-ru", "en-tr", "en-zh"]
wmt17_sys_lang_pairs = wmt17_sys_to_lang_pairs + wmt17_sys_from_lang_pairs

import argparse


def get_wmt17_sys_data(lang_pair):
    first, second = lang_pair.split("-")

    human_scores = pd.read_csv("wmt17/manual-evaluation/DA-syslevel.csv", delimiter=" ")

    with open(
        "wmt17/input/wmt17-metrics-task/"
        "wmt17-submitted-data/txt/references/newstest2017-{}{}-ref.{}".format(
            first, second, second
        )
    ) as f:
        refs = f.read().strip().split("\n")

    gold_dict = dict(
        zip(
            human_scores[human_scores["LP"] == lang_pair]["SYSTEM"],
            human_scores[human_scores["LP"] == lang_pair]["HUMAN"],
        )
    )
    gold_scores = []

    lang_dir = (
        "wmt17/input/"
        "wmt17-metrics-task/wmt17-submitted-data/"
        "txt/system-outputs/newstest2017/{}".format(lang_pair)
    )
    systems = [system[13:-6] for system in os.listdir(lang_dir)]

    refs *= len(systems)
    cands = []

    for system in systems:

        with open(
            os.path.join(lang_dir, "newstest2017.{}.{}".format(system, lang_pair))
        ) as f:
            cand_sys = f.read().strip().split("\n")
        gold_scores.append(gold_dict[system])

        cands += cand_sys
    return refs, cands, gold_scores, systems


def get_wmt17_sys_bert_score(
    lang_pair, scorer, cache=False, from_en=True, batch_size=64
):
    filename = ""
    if from_en:
        if scorer.idf:
            filename = "cache_score/from_en/17/{}/wmt17_seg_from_{}_{}_idf.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
        else:
            filename = "cache_score/from_en/17/{}/wmt17_seg_from_{}_{}.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
    else:
        if scorer.idf:
            filename = "cache_score/to_en/17/{}/wmt17_seg_to_{}_{}_idf.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )
        else:
            filename = "cache_score/to_en/17/{}/wmt17_seg_to_{}_{}.pkl".format(
                scorer.model_type, *lang_pair.split("-")
            )

    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pkl.load(f)
    else:
        refs, cands, gold_scores, systems = get_wmt17_sys_data(lang_pair)
        if scorer.idf:
            scorer.compute_idf(refs)
        raw_scores = scorer.score(cands, refs, batch_size=batch_size)
        scores = [s.view(len(systems), -1).mean(dim=-1) for s in raw_scores]

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pkl.dump((scores, gold_scores), f)

    return scores, gold_scores


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
        default=wmt17_sys_to_lang_pairs,
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
            scores, gold_scores = get_wmt17_sys_bert_score(
                lang_pair, scorer, batch_size=args.batch_size, cache=True, from_en=False
            )
            for s, name in zip(scores, ["P", "R", "F"]):
                results[lang_pair][f"{model_type} {name}"] = np.mean(
                    pearsonr(gold_scores, s)[0]
                )

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
