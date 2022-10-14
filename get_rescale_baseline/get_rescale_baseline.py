import argparse
import gzip
import os
from random import shuffle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacrebleu
import torch
from tqdm.auto import tqdm

import bert_score


def get_data(lang="en"):

    if lang == "en":
        file_path = "data/news.2017.en.shuffled.deduped"
    elif lang == "zh":
        file_path = "data/paracrawl/crawl_chinese.txt"
    else:
        file_path = f"data/paracrawl/rand_{lang}.txt"

    with open(file_path, "r") as f:
        lines = []
        for i, line in enumerate(f):
            if i == 1_000_000:
                break
            line = line.strip()
            if len(line.split(" ")) < 32 and len(line.split(" ")) > 0:
                lines.append(line)

    samples = np.random.choice(
        range(len(lines)), size=(2, len(lines) // 2), replace=False
    )

    hyp = [lines[i] for i in samples[0]]
    cand = [lines[i] for i in samples[1]]

    return hyp, cand


def chunk(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--lang", type=str, required=True, help="language to compute baseline with"
    )
    parser.add_argument("-m", "--model", nargs="+", help="models to tune")
    parser.add_argument("-b", "--batch_size", type=int, default=64)

    args = parser.parse_args()

    hyp, cand = get_data(lang=args.lang)

    for model_type in args.model:
        baseline_file_path = f"rescale_baseline/{args.lang}/{model_type}.tsv"
        if os.path.isfile(baseline_file_path):
            print(f"{model_type} baseline exists for {args.lang}")
            continue
        else:
            print(f"computing baseline for {model_type} on {args.lang}")
            scorer = bert_score.BERTScorer(model_type=model_type, all_layers=True)
            with torch.no_grad():
                score_means = None
                count = 0
                for batches in tqdm(
                    chunk(list(zip(hyp, cand)), 1000), total=len(hyp) / 1000
                ):
                    batch_hyp, batch_cand = zip(*batches)
                    scores = scorer.score(
                        batch_hyp, batch_cand, batch_size=args.batch_size
                    )
                    scores = torch.stack(scores, dim=0)
                    if score_means is None:
                        score_means = scores.mean(dim=-1)
                    else:
                        score_means = score_means * count / (
                            count + len(batches)
                        ) + scores.mean(dim=-1) * len(batches) / (count + len(batches))
                    count += len(batches)

            pd_baselines = pd.DataFrame(
                score_means.numpy().transpose(), columns=["P", "R", "F"]
            )
            pd_baselines.index.name = "LAYER"

            os.makedirs(os.path.dirname(baseline_file_path), exist_ok=True)
            pd_baselines.to_csv(baseline_file_path)
            del scorer
