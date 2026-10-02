"""Segment-level frozen speech embeddings for multiple-instance training.

Each clip is cut into SEG-second segments (hop HOP s); every segment gets its own embedding:
  WavLM-large: mean over time of the average of layers 12-20 (1024-d)
  Whisper encoder: mean over the segment's real frames of the average of layers 20-32 (1280-d)
Saves cache/seg_<model>.npz with emb (n_segments, d) and clip (n_segments,) = row index into the clip table.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from asr_transcribe import DATA, ROOT, SR, load_audio


def segments(y, seg, hop):
    n = len(y)
    if n <= seg * SR:
        return [y]
    starts = list(range(0, n - seg * SR + 1, hop * SR))
    if starts[-1] + seg * SR < n:                       # make sure the tail is covered
        starts.append(n - seg * SR)
    return [y[s:s + seg * SR] for s in starts]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["wavlm_large", "whisper"], default="wavlm_large")
    ap.add_argument("--seg", type=int, default=10)
    ap.add_argument("--hop", type=int, default=5)
    a = ap.parse_args()
    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    if a.model == "wavlm_large":
        from transformers import AutoFeatureExtractor, AutoModel
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        m = AutoModel.from_pretrained("microsoft/wavlm-large", dtype=torch.float16).cuda().eval()
        def emb(c):
            x = fe(c, sampling_rate=SR, return_tensors="pt").input_values.cuda().half()
            hs = m(x, output_hidden_states=True).hidden_states
            return torch.stack(hs[12:21]).mean(0)[0].float().mean(0).cpu().numpy()
    else:
        from transformers import WhisperFeatureExtractor, WhisperModel
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
        m = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo", dtype=torch.float16).encoder.cuda().eval()
        def emb(c):
            x = fe(c, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
            n = max(1, int(len(c) / SR * 50))
            hs = m(x, output_hidden_states=True).hidden_states
            return torch.stack(hs[20:33]).mean(0)[0, :n].float().mean(0).cpu().numpy()
    E, C = [], []
    for i, (s, f) in enumerate(zip(meta.split, meta.filename)):
        for c in segments(load_audio(os.path.join(DATA, s, f)), a.seg, a.hop):
            E.append(emb(c)); C.append(i)
        if i % 100 == 0:
            print(i, flush=True)
    np.savez(os.path.join(ROOT, "cache", f"seg_{a.model}_{a.seg}s.npz"), emb=np.stack(E).astype(np.float32), clip=np.array(C))
    print("done", len(E))


if __name__ == "__main__":
    main()
