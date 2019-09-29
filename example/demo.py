from bert_score import score

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

P, R, F = score(cands, refs, lang='en')
print(f'P={P:.6f} R={R:.6f} F={F:.6f}')
