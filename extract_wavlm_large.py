"""Frozen WavLM-large embeddings: per-layer mean pooling over 20 s windows.

Saves cache/wavlm_large_layers.npy with shape (n_clips, 25, 1024). Self-supervised model, used as a fixed
feature extractor only.
"""
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

from asr_transcribe import DATA, ROOT, SR, load_audio

MODEL = "microsoft/wavlm-large"
OUT = os.path.join(ROOT, "cache", "wavlm_large_layers.npy")


@torch.no_grad()
def main(win_s=20):
    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    out = np.zeros((len(meta), 25, 1024), np.float32)
    for i, (s, f) in enumerate(zip(meta.split, meta.filename)):
        y = load_audio(os.path.join(DATA, s, f))
        hs = []
        for a in range(0, len(y), win_s * SR):
            c = y[a:a + win_s * SR]
            if len(c) < SR and a > 0:
                continue
            x = fe(c, sampling_rate=SR, return_tensors="pt").input_values.cuda().half()
            hs.append(torch.stack(m(x, output_hidden_states=True).hidden_states).squeeze(1).float().cpu())
        out[i] = torch.cat(hs, 1).mean(1).numpy()
        if i % 100 == 0:
            print(i, flush=True)
    np.save(OUT, out)
    print("done", out.shape)


if __name__ == "__main__":
    main()
