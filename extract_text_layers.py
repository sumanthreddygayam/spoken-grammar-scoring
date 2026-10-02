"""Frozen per-layer text embeddings (mean-pooled) from self-supervised LMs, for layer selection.

Saves cache/text_layers_<model>.npy with shape (n_clips, n_layers+1, hidden).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from asr_transcribe import DATA, ROOT

FILLER_RE = re.compile(r"\b(?:u+[hm]+|e+rm*|a+h+|h+m+|m+h*m+|uh-huh)\b[,.]?", re.I)


def clean(t):
    t = FILLER_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip(" ,.")


@torch.no_grad()
def embed(name, texts, max_len=512):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModel.from_pretrained(name).cuda().eval()
    out = []
    for t in texts:
        enc = tok(t or ".", truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        hs = m(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0].mean(0) for h in hs]).float().cpu().numpy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="transcripts.csv")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    ap.add_argument("--models", nargs="+", default=["roberta-base", "openai-community/gpt2"])
    args = ap.parse_args()
    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    tr = pd.read_csv(os.path.join(ROOT, "cache", args.transcripts), keep_default_na=False)
    texts = meta.merge(tr, on=["split", "filename"], how="left").transcript.fillna("").map(clean).tolist()
    for name in args.models:
        arr = embed(name, texts, 512 if "roberta" in name else 1024)
        np.save(os.path.join(ROOT, "cache", f"text_layers_{name.split('/')[-1]}{args.suffix}.npy"), arr)
        print(name, arr.shape, flush=True)


if __name__ == "__main__":
    main()
