## 10. Report

### 10.1 Approach
Grammar is a property of *what* is said, but proficiency also shows in *how* it is said. The engine converts each
recording into several complementary representations and learns, from the 732 rubric-scored training clips only,
how each relates to the 0–5 grammar score:

1. **Audio → transcript** with Whisper large-v3-turbo (frozen; long-form decoding with temperature fallback).
2. **Transcript → linguistic evidence**:
   * 43 interpretable text features: sentence structure (length, dependency depth, subordinate clauses, fragments),
     error heuristics (subject–verb agreement, *a/an*, doubled words), tense and aspect use, lexical diversity,
     POS mix, and GPT-2 surprisal.
   * RoBERTa-base (layer 10) and **Qwen2.5-1.5B / 3B base LMs** (layers 16–24 / 18–30) contextual embeddings, with layers chosen by CV.
3. **Audio → speech evidence**: pause and prosody features, plus three frozen speech encoders. These are WavLM-base,
   WavLM-large (layers 12–20) and the Whisper encoder (layers 20–32), time-averaged per clip. WavLM-large is
   *also* used per 10 s segment for a multiple-instance SVR.
4. **Regression**: one model per representation (Ridge, SVR, LightGBM), plus a **joint multi-block kernel SVR**
   over all encoders, combined by a non-negative linear stack. Predictions are clipped to [0, 5].

**Data policy.** The only supervision is the provided training labels. No external grammar-labelled data is used,
and no network is fine-tuned. Pretrained models act purely as fixed feature extractors. The test set is used only
to write `submission.csv`.

### 10.2 Preprocessing
* Decoding to mono 16 kHz float32, peak-normalised. Pauses are preserved.
* Adaptive energy VAD for fluency features. The threshold is relative to each clip's own noise floor, so
  noisy recordings are handled.
* Transcript cleaning removes hesitation fillers (counted first, as a feature) and ASR repetition loops, and
  **never corrects grammar**.
* **Label audit:** the 37 clips labelled 0 (outside the 1–5 rubric) form a separate batch: file IDs 5037–5073,
  near-flat audio with ≈4 dB dynamic range against ≈44 dB for the others, and none in the test set. They are
  excluded from training. The ablation in Section 7 shows that keeping them worsens CV RMSE on rubric-scored clips
  (Ridge 0.747 → 0.808, LightGBM 0.720 → 0.737).

### 10.3 Step-by-step analysis: what moved the RMSE

| Step / change | CV RMSE | Effect |
|---|---|---|
| Hand-crafted features only (best: LightGBM) | 0.720 | baseline of explicit grammar evidence |
| + RoBERTa text embedding (last layer) | 0.638 alone | text representation beats hand-made features |
| + WavLM-base audio embedding | 0.541 alone | **audio is the strongest single source** |
| First stacked ensemble (v1) | 0.491 | complementary views combine well |
| Layer selection per encoder (Section 7a) | RoBERTa 0.639 → 0.625 | upper-middle layers beat the last layer |
| + WavLM-large L12–20, Whisper encoder L20–32 | 0.529 / 0.537 alone | stronger, partly complementary audio views |
| + joint multi-block kernel SVR | **0.489 alone** | best single model: audio × text interactions |
| Stacked ensemble (v2) | 0.479 | −0.013 vs v1 (leaderboard 0.369 → 0.3555) |
| + Qwen2.5-1.5B base LM text embedding, layers 16–24 | 0.604 alone (vs RoBERTa 0.625) | a larger LM reads grammar better |
| **Final stacked ensemble (v3)** | **0.471** | −0.008 vs v2; Qwen becomes the most important member |
| Fine-tuning WavLM-base end-to-end (5-fold, train only) | 0.596 alone, no stack gain | dropped: 732 clips are too few to beat frozen features |
| Qwen inside the joint kernel SVR; isotonic / SVR / LightGBM stackers | no gain | dropped |
| **Length-aware blending** (separate weights for 45 s and 60 s clips) | **0.4685**; test-mix 0.5086 → **0.5059** | audio encoders are weak on 45 s clips, and 69 % of test clips are 45 s |
| Zero-shot LLM judge (Qwen2.5-7B-instruct, rubric prompt, expected score from digit log-probs) | r = 0.44 alone, no stack gain | dropped |
| Clip-length indicators inside the models; more Qwen/RoBERTa text variants | ≤ 0.003 | dropped |
| **Segment-level WavLM-large SVR** (10 s segments, clip = mean of segments) | CV 0.4691 → 0.4648 | ~11× more training examples for the kernel model |
| **+ Qwen2.5-3B base LM** (layers 18–30) | **CV 0.4626**, test-mix **0.5034** | a second, larger text view |
| Segment-level Whisper encoder | no gain | dropped |
| Pseudo-labelling with unlabelled test audio (leak-free fold-wise test) | slightly worse | dropped; the final model uses training labels only |
| Snapping 45 s predictions to whole numbers | slightly worse | dropped |
| **SVR hyperparameter tuning** per encoder (C, ε, γ grid by CV) | every SVR improves alone (e.g. WavLM-base 0.541 → 0.524) | kept |
| Verbatim re-transcription (Whisper prompted to keep fillers, false starts and errors; fallback to the standard transcript for 11 degenerate outputs) | text models worse: RoBERTa 0.611 → 0.657, Qwen-1.5B 0.602 → 0.616 | dropped. The extra disfluencies swamp the embeddings; the standard transcripts are kept |
| Grammar-prompted LLM embedding (final-token state after a grammar question) | no gain on standard transcripts | dropped |

**How much of a CV difference is real?** Re-running the *same* models with a different assignment of clips to
folds moved the stacked test-mix RMSE by up to ≈ 0.004. Changes smaller than that are treated as noise, and the
final model only includes components that improved by more.
| Per-layer WavLM-base pooling, mean+std pooling, GPT-2 embeddings, extra stack members | no gain (≤ 0.001) | tested and dropped |
| Rounding predictions to the 0.5 label grid | 0.479 → 0.501 | **worse**, so not used for the main submission |

The ablation (Section 7c) shows that audio encoders are the largest group, while the single most valuable member is
the Qwen text model (+0.008 RMSE when removed). The hand-crafted block adds little on top of the encoders, though it
is the most interpretable part.

### 10.4 Final evaluation

| Final ensemble | RMSE | Pearson |
|---|---|---|
| Training RMSE (in-sample, 732 rubric-scored clips) | 0.157 | see Section 8 |
| Cross-validated (out-of-fold, 5-fold × 3 repeats) | 0.4626 | see Section 7 |
| Test-mix CV (re-weighted to the test's 69 % short clips) | 0.5034 | n/a |
| Public leaderboard: v1 / v2 / v3 / v4 (length-aware) / v4 snapped | 0.369 / 0.3555 / 0.3596 / 0.3464 / 0.3457 | n/a |

**Why v3 scored slightly worse publicly despite better CV:** an RMSE measured on ≤ 216 clips has a sampling error of
roughly ±0.02 at this error level. Changes of a few thousandths on the public leaderboard are within that noise.
The final model is chosen on CV and test-mix CV, which average over 732 clips × 3 repeats, so that it does not
overfit the public split.

The in-sample training RMSE is required by the brief. It is much lower than the CV figure because SVRs fit their
training set closely. **Cross-validated RMSE is the honest estimate.** It has tracked the leaderboard closely: v1 → v2 improved CV by 0.013 and the leaderboard by 0.0135. The test clips
appear easier than the average training fold, so absolute leaderboard scores sit about 0.12 below CV.

**On rounding to the label grid.** Snapping to 0.0, 0.5, …, 5.0 raises RMSE (0.471 → 0.495 on OOF predictions).
The simulation in Section 7c shows rounding only pays off once raw errors are already below about 0.15.
`submission_rounded.csv` is written for comparison, but `submission.csv` (continuous) is the recommended entry.

### 10.5 Interpretation
* **Learned speech representations carry most of the signal.** Human grammar ratings correlate strongly with
  overall fluency and proficiency, which the speech encoders capture. The Whisper encoder also "hears"
  hesitations and false starts that the cleaned transcript loses.
* **Among explicit features**, lexical range (Guiraud index, r ≈ 0.53), speech volume, fewer pauses and lower GPT-2
  surprisal go with higher scores. Run-on "sentences" chained with *and*, and doubled words, go with lower scores.
  This matches the rubric.
* **Clip length matters.** On 45 s clips the speech encoders lose about 0.12 RMSE, while the text models do not. The
  length-aware blender responds by weighting text (Qwen, RoBERTa) and hand-crafted features more on short clips, and
  the speech encoders more on long clips.
* **Errors** concentrate at the extremes: clips rated 1–2 are over-predicted and some 5s under-predicted
  (shrinkage toward the mean, with only 4 training clips below 2.0).

### 10.6 Limitations and next steps
* **No speaker IDs**, so CV cannot be speaker-grouped.
* Whisper normalises some disfluencies and errors. A verbatim ASR would preserve more grammatical evidence.
* End-to-end fine-tuning of the speech encoders would need more labelled data than 732 clips to beat the
  frozen-feature approach reliably.
* Human ratings on a half-point grid carry rater noise, which puts a floor under any honest held-out RMSE.
  Held-out RMSE below 0.2 would require predicting raters' individual half-point choices almost exactly. At that
  point the model would be fitting rater noise rather than grammar.
