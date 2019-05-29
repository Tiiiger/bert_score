import numpy as np

from bert_score import score
from bert_score.utils import precompute_sen_embeddings

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

P, R, F = score(cands, refs, bert="bert-base-uncased")

# Pre-compute sentence embeddings, then rescore

sen_to_embedding, idf_dict = precompute_sen_embeddings(
    cands + refs, bert="bert-base-uncased")


P_pc, R_pc, F_pc = score(cands, refs,
                         sen_to_embedding=sen_to_embedding,
                         idf_dict=None,
                         bert="bert-base-uncased")

assert np.allclose(P, P_pc)
assert np.allclose(R, R_pc)
assert np.allclose(F, F_pc)
