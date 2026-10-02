"""Stage 2 (ASR): transcribe every train/test clip with Whisper and cache the result.

Run once (it is the slowest stage); the notebook reads cache/transcripts.csv.
Uses Whisper's native long-form decoding with temperature fallback, which suppresses
the repetition loops that chunked greedy decoding produces on noisy clips.
"""
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "kaggle-dataset", "Dataset_Final")
OUT = os.path.join(ROOT, "cache", "transcripts.csv")
SR = 16_000
MODEL = "openai/whisper-large-v3-turbo"
# Verbatim mode: a prompt full of fillers, false starts and a grammar error ("he go") steers Whisper away from
# its habit of tidying speech, so learners' grammatical errors are kept in the transcript.
VERBATIM_PROMPT = "Umm, so, uh, I- I think that, like, he go to the market and, um, buy some, uh, vegetables. Yeah."


def load_audio(path):
    """Mono, 16 kHz, peak-normalised float32. Internal pauses are kept."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.abs(y).max()
    return (y / peak * 0.95).astype(np.float32) if peak > 0 else y


class Transcriber:
    def __init__(self, device="cuda", verbatim=False):
        self.device, self.verbatim = device, verbatim
        self.proc = WhisperProcessor.from_pretrained(MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
        self.prompt_ids = self.proc.get_prompt_ids(VERBATIM_PROMPT, return_tensors="pt").to(device) if verbatim else None

    @torch.no_grad()
    def __call__(self, y):
        feats = self.proc(y, sampling_rate=SR, return_tensors="pt", truncation=False,
                          padding="longest", return_attention_mask=True)
        if feats.input_features.shape[-1] < 3000:  # short clip: pad to Whisper's 30 s window
            feats = self.proc(y, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
        ids = self.model.generate(
            feats.input_features.to(self.device, torch.float16),
            attention_mask=feats.attention_mask.to(self.device),
            language="en", task="transcribe", return_timestamps=True,
            condition_on_prev_tokens=self.verbatim,
            **({"prompt_ids": self.prompt_ids, "prompt_condition_type": "all-segments"} if self.verbatim else {}),
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.8, logprob_threshold=-1.0, no_speech_threshold=0.6,
        )
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main(limit=None, verbatim=False):
    out_path = OUT.replace(".csv", "_verbatim.csv") if verbatim else OUT
    items = []
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(DATA, f"{split}.csv"))
        items += [(split, f) for f in df.filename]

    done = pd.read_csv(out_path, keep_default_na=False) if os.path.exists(out_path) else pd.DataFrame(columns=["split", "filename", "transcript"])
    seen = set(zip(done.split, done.filename))
    todo = [it for it in items if it not in seen][:limit]
    print(f"{len(todo)} files to transcribe ({len(seen)} cached)", flush=True)

    asr, rows, t0 = Transcriber(verbatim=verbatim), [], time.time()
    for i, (split, fname) in enumerate(todo, 1):
        rows.append({"split": split, "filename": fname, "transcript": asr(load_audio(os.path.join(DATA, split, fname)))})
        if i % 25 == 0 or i == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out_path, index=False)
            rows = []
            print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/file", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verbatim"]
    main(int(args[0]) if args else None, verbatim="--verbatim" in sys.argv)
