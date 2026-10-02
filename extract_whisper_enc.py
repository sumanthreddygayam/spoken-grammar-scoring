"""Frozen Whisper-encoder embeddings: per-layer mean+std pooling over real (non-padded) frames.

Saves cache/whisper_enc_layers.npy with shape (n_clips, n_layers_kept, 2, 1280).
Whisper is used only as a fixed feature extractor; nothing is trained here.
"""
import os

import numpy as np
import pandas as pd
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

from asr_transcribe import DATA, ROOT, SR, load_audio

MODEL = "openai/whisper-large-v3-turbo"
LAYERS = list(range(4, 33, 4))          # hidden_states index 4, 8, ..., 32
OUT = os.path.join(ROOT, "cache", "whisper_enc_layers.npy")


@torch.no_grad()
def main():
    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    enc = WhisperModel.from_pretrained(MODEL, dtype=torch.float16).encoder.cuda().eval()
    out = np.zeros((len(meta), len(LAYERS), 2, 1280), np.float32)
    for i, (s, f) in enumerate(zip(meta.split, meta.filename)):
        y = load_audio(os.path.join(DATA, s, f))
        hs = []
        for a in range(0, len(y), 30 * SR):                 # Whisper's fixed 30 s window
            c = y[a:a + 30 * SR]
            if len(c) < SR and a > 0:
                continue
            x = fe(c, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
            n = max(1, int(len(c) / SR * 50))               # encoder frames are 20 ms; drop padding
            h = enc(x, output_hidden_states=True).hidden_states
            hs.append(torch.stack([h[l][0, :n] for l in LAYERS]).float().cpu())
        h = torch.cat(hs, 1)                                # layers, T, 1280
        out[i, :, 0], out[i, :, 1] = h.mean(1).numpy(), h.std(1).numpy()
        if i % 100 == 0:
            print(i, flush=True)
    np.save(OUT, out)
    print("done", out.shape)


if __name__ == "__main__":
    main()
