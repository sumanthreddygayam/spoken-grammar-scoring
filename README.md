# Spoken Grammar Scoring Engine

Predicts a 0–5 grammar score (MOS Likert, human-rated) for 45–60 s spoken English answers, from audio alone.
Built for a Kaggle competition with 769 labelled training clips and 216 test clips.

**Main deliverable:** [`grammar_scoring.ipynb`](grammar_scoring.ipynb). It is fully executed, and contains the
pipeline, every evaluation, the visualisations, and a written report (Section 10).

## Results

| | RMSE | Pearson r |
|---|---|---|
| Cross-validated (5-fold × 3 repeats, out-of-fold) | **0.460** | 0.89 |
| Public leaderboard | **0.3399** | n/a |
| Training fit (in-sample) | 0.16 | 0.99 |

## Approach

```
audio ─► preprocess (mono, 16 kHz, peak-normalised, pauses kept)
      ├► Whisper large-v3-turbo ─► transcript ─► cleaning (fillers/ASR loops removed, grammar never corrected)
      │        ├► hand-crafted grammar features (spaCy syntax, agreement checks, tense, lexical diversity, GPT-2 surprisal)
      │        └► frozen text LMs: RoBERTa-base, Qwen2.5-1.5B / 3B (base) – layers chosen by CV
      └► frozen speech encoders: WavLM-base, WavLM-large (clip-level and 10 s segments), Whisper encoder
                 │
      per-block regressors (tuned SVR / Ridge / LightGBM) + joint multi-block kernel SVR
                 │
      length-aware stacked ensemble (separate blend weights for 45 s and 60 s clips) ─► clip to [0, 5]
```

All pretrained networks are used **frozen, as feature extractors only**. The only supervision is the competition's
training labels. No external grammar-labelled data is used, and the test set is never used for any modelling choice.

## Key findings

- **Label audit:** the 37 clips labelled 0 (outside the 1–5 rubric) form a separate recording batch with near-flat
  audio and no counterpart in the test set. Excluding them improves validation RMSE on rubric-scored clips.
- **Speech encoders carry most of the signal.** Removing all audio models costs about +0.05 RMSE. Grammar ratings
  correlate strongly with overall fluency.
- **Clip length matters.** Audio models are much weaker on 45 s clips, which make up 69 % of the test set. A
  length-aware blender shifts weight to the text models for short clips.
- **Cross-validation tracks the leaderboard.** Every CV improvement that exceeded fold noise (≈ 0.004) also improved
  the public score. A "test-mix" CV, re-weighted to the test's clip-length mix, was used for model selection.
- **Negative results, documented in the report:** pseudo-labelling, end-to-end fine-tuning of WavLM, a zero-shot
  7B LLM judge, verbatim (disfluency-preserving) transcription, output calibration and rounding to the label grid
  did not help.

## Repository layout

| File | Purpose |
|---|---|
| `grammar_scoring.ipynb` | Executed notebook: pipeline, evaluation, plots, report |
| `build_notebook.py` | Generates the notebook from source (keeps it diff-able) |
| `report_section.md` | Report text included as the notebook's final section |
| `asr_transcribe.py` | Whisper transcription (standard and verbatim modes), resumable |
| `extract_*.py` | Frozen feature extraction (Whisper encoder, WavLM-large, segments, text LMs) |
| `finetune_wavlm.py`, `llm_judge.py` | Experiments that were tested and not kept |

## Reproducing

1. Place the competition data in `kaggle-dataset/Dataset_Final/` (`train.csv`, `test.csv`, `train/`, `test/`).
2. Install the dependencies from `requirements.txt` (CUDA PyTorch first), plus `python -m spacy download en_core_web_sm`.
3. `python build_notebook.py`, then run `grammar_scoring.ipynb`. Expensive steps (ASR, embeddings) are cached
   in `cache/`. A first run takes a few hours on a 6 GB GPU; later runs take minutes.

Hardware used: a single NVIDIA RTX 3050 laptop GPU (6 GB).
