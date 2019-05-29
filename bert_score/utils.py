import time
import torch
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel


__all__ = ['bert_types']

bert_types = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
]

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=False)
    return x_encoded_layers


def process(a, tokenizer=None, max_len=500):
    if not tokenizer is None:
        a = ["[CLS]"] + tokenizer.tokenize(a)[:max_len - 2] + ["[SEP]"]
        a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer, max_len=tokenizer.max_len)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict


def collate_idf(arr, tokenize, numericalize, idf_dict, max_len,
                pad="[PAD]", device='cuda:0'):
    """
    Helper function that pads a list of sentences to hvae the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [["[CLS]"] + tokenize(a)[:max_len - 2] + ["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       sen_to_embedding=None, batch_size=-1,
                       device='cuda:0', pad="[PAD]"):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `sen_to_embedding` (dict): a map of string->bert_embedding
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    padded_sens, padded_idf, lens, mask = collate_idf(all_sens,
                                                      tokenizer.tokenize,
                                                      tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      max_len=tokenizer.max_len,
                                                      device=device)

    if not sen_to_embedding:
        if batch_size == -1: batch_size = len(all_sens)

        # Compute indices of unique sens
        unique_sens, unique_idxs = set(), []
        for i, sen in enumerate(all_sens):
            if sen not in unique_sens:
                unique_idxs.append(i)
            unique_sens.add(sen)

        # Embed each unique sen
        sen_to_embedding = {}
        with torch.no_grad():
            for i in range(0, len(unique_sens), batch_size):
                idxs = unique_idxs[i:i+batch_size]
                batch_embedding = bert_encode(model, padded_sens[idxs], attention_mask=mask[idxs])
                for idx, embed in zip(idxs, batch_embedding):
                    sen = all_sens[idx]
                    sen_to_embedding[sen] = embed[:lens[idx]].cpu()  # Trim embedding to original sequence length

    embeddings = []
    max_seq_len = padded_sens.size(-1)
    for sen, sen_len in zip(all_sens, lens):
        embed = torch.FloatTensor(sen_to_embedding[sen])
        seq_len, embed_dim = embed.size(0), embed.size(-1)
        padding = torch.ones(max_seq_len - seq_len, embed_dim)
        embed = torch.cat([embed, padding])
        embeddings.append(embed)
    total_embedding = torch.stack(embeddings)
    return total_embedding, lens, mask, padded_idf


def greedy_cos_idf(ref_embedding, ref_lens, ref_masks, ref_idf,
                   hyp_embedding, hyp_lens, hyp_masks, hyp_idf):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
    """

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, masks.size(1), masks.size(2))\
                              .contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)

    F = 2 * P * R / (P + R)
    return P, R, F


def bert_cos_score_idf(model, refs, hyps, tokenizer, idf_dict,
                       sen_to_embedding=None, verbose=False,
                       batch_size=64, device='cuda:0', ipynb_mode=False):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `sen_to_embedding` (dict): a map of string->bert_embedding
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    preds = []
    iter_range = range(0, len(refs), batch_size)
    if verbose and not ipynb_mode:
        iter_range = tqdm(iter_range)
    for batch_start in iter_range:
        if verbose and ipynb_mode and batch_start % 100 == 0:
            print(f'Batch: {batch_start / batch_size} / {(len(refs) / batch_size)}')
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        ref_stats = get_bert_embedding(batch_refs, model, tokenizer,
                                       idf_dict=idf_dict,
                                       sen_to_embedding=sen_to_embedding,
                                       device=device)
        hyp_stats = get_bert_embedding(batch_hyps, model, tokenizer,
                                       idf_dict=idf_dict,
                                       sen_to_embedding=sen_to_embedding,
                                       device=device)
        P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats)
        preds.append(torch.stack((P, R, F1), dim=1).cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def precompute_sen_embeddings(sens, bert="bert-base-multilingual-cased",
                              num_layers=8, verbose=False, no_idf=False,
                              batch_size=64, get_idf_dict_nthreads=1,
                              state_dict=None):
    """
    Precompute BERT embeddings for all sentences in `sens`.

    Args:
        - :param: `sens` (list of str)
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
            when composing the embeddings
        - :param: `no_idf` (bool): do not use idf weighting
        - :param: `get_idf_dict_nthreads` (int): number of threads to use
            to comose the idf_dict
        - :param: `state_dict`: optionally pass in a PyTorch state dict
            with which to instantiate the bert model
    """
    assert bert in bert_types

    tokenizer = BertTokenizer.from_pretrained(bert)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'loading {bert} model...')
    model = BertModel.from_pretrained(bert, state_dict=state_dict)
    model.eval()
    model.to(device)
    # drop unused layers
    model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

    if no_idf:
        idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[101] = 0
        idf_dict[102] = 0
    else:
        if verbose:
            print(f'preparing IDF dict with {get_idf_dict_nthreads} threads...')
        start = time.perf_counter()
        idf_dict = get_idf_dict(sens, tokenizer, get_idf_dict_nthreads)
        if verbose:
            print('done in {:.2f} seconds'.format(time.perf_counter() - start))

    # Compute BERT embeddings
    sen_to_embedding = {}
    sens = list(set(sens))
    iter_range = range(0, len(sens), batch_size)
    if verbose: iter_range = tqdm(iter_range)
    for batch_start in iter_range:
        batch_sens = sens[batch_start:batch_start+batch_size]
        total_embedding, lens, mask, padded_idf = get_bert_embedding(batch_sens, model, tokenizer, idf_dict, device=device)
        for sen, embedding, sen_len in zip(batch_sens, total_embedding, lens):
            sen_to_embedding[sen] = embedding[:sen_len].numpy()
    return sen_to_embedding, idf_dict
