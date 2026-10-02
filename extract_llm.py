"""Frozen base-LLM features for each transcript (Qwen2.5-1.5B base: self-supervised next-token pre-training only).

Saves cache/llm_<tag>.npz with
  layers: (n_clips, n_layers+1, hidden) mean-pooled hidden states of every layer
  nll:    (n_clips, 6) token surprisal statistics: mean, std, p90, max, share of tokens with NLL > 5, > 8
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr_transcribe import DATA, ROOT

FILLER_RE = re.compile(r"\b(?:u+[hm]+|e+rm*|a+h+|h+m+|m+h*m+|uh-huh)\b[,.]?", re.I)
LOOP_RE = re.compile(r"\b(.{4,80}?)(?:[\s,.]+\1\b){3,}", re.I)


def clean(t):
    # Same cleaning as the notebook: drop ASR loops and fillers, never touch grammar.
    t = LOOP_RE.sub(r"\1", t)
    t = FILLER_RE.sub(" ", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"([,.!?])[,.]+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip(" ,.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tag", default="qwen15")
    ap.add_argument("--transcripts", default="transcripts.csv", help="cache file with the transcripts to embed")
    ap.add_argument("--prompted", action="store_true",
                    help="wrap the transcript in a grammar-focused instruction and also keep the last-token state")
    args = ap.parse_args()
    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    tr = pd.read_csv(os.path.join(ROOT, "cache", args.transcripts), keep_default_na=False)
    texts = meta.merge(tr, on=["split", "filename"], how="left").transcript.fillna("").map(clean).tolist()

    tok = AutoTokenizer.from_pretrained(args.model)
    # device_map="auto" spills layers to CPU RAM when the model does not fit in 6 GB of VRAM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto",
                                             max_memory={0: "5GiB", "cpu": "12GiB"}).eval()
    layers, last, nll = [], [], []
    for i, t in enumerate(texts):
        ids = tok((tok.bos_token or "") + (t or "."), return_tensors="pt").input_ids[:, :1024].to(m.device)
        out = m(ids, output_hidden_states=True)
        layers.append(torch.stack([h[0, 1:].float().mean(0) for h in out.hidden_states]).cpu().numpy())
        if args.prompted:
            # Grammar-focused prompt; the final token's hidden state summarises the transcript "for" this question.
            p = ("Below is a transcript of a learner's spontaneous spoken English, including hesitations and errors.\n"
                 f'Transcript: "{t}"\n'
                 "Question: How grammatically accurate is this speaker (sentence structure, verb forms, agreement)?\n"
                 "Answer: The speaker's grammar is")
            pids = tok(p, return_tensors="pt").input_ids[:, -1500:].to(m.device)
            ph = m(pids, output_hidden_states=True).hidden_states
            last.append(torch.stack([h[0, -1].float() for h in ph]).cpu().numpy())
        lp = torch.log_softmax(out.logits[0, :-1].float(), -1)
        tok_nll = -lp.gather(1, ids[0, 1:, None]).squeeze(1).cpu().numpy()
        if tok_nll.size == 0:                    # transcript reduced to one token (e.g. only fillers): nothing to score
            tok_nll = np.zeros(1)
        nll.append([tok_nll.mean(), tok_nll.std(), np.percentile(tok_nll, 90), tok_nll.max(),
                    (tok_nll > 5).mean(), (tok_nll > 8).mean()])
        if i % 200 == 0:
            print(i, flush=True)
    extra = {"last": np.stack(last).astype(np.float16)} if args.prompted else {}
    np.savez(os.path.join(ROOT, "cache", f"llm_{args.tag}.npz"), layers=np.stack(layers).astype(np.float16), nll=np.array(nll), **extra)
    print("done", np.stack(layers).shape)


if __name__ == "__main__":
    main()
