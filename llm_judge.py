"""Zero-shot LLM grammar judge (frozen Qwen2.5-7B-instruct served by a local Ollama).

For each transcript the model is shown the competition's 1-5 grammar rubric and asked for one digit. Instead of
taking its answer, we read the probability it assigns to each digit (first-token log-probs) and keep:
expected score, p(1..5), and entropy. These become *features*; the regressors that map them to the label are
trained only on our training labels. Nothing is trained here.

Saves cache/judge_<tag>.csv (split, filename, judge_* columns). Resumable.
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd

from extract_llm import clean
from asr_transcribe import DATA, ROOT

RUBRIC = """Grammar score rubric:
1 - Struggles with proper sentence structure and syntax; limited control over simple grammatical structures and memorized sentence patterns.
2 - Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic sentence structure and grammatical mistakes; may leave sentences incomplete.
3 - Decent grasp of sentence structure but makes errors in grammatical structure, or decent grasp of grammatical structure but errors in sentence syntax and structure.
4 - Strong understanding of sentence structure and syntax; consistently good control of grammar. Occasional minor errors that do not lead to misunderstandings; can correct most of them.
5 - High grammatical accuracy and adept control of complex grammar; uses grammar accurately and effectively, seldom making noticeable mistakes; handles complex structures well and self-corrects when necessary."""

PROMPTS = {
    "v1": ("You are an experienced examiner of spoken English. You rate only grammar, not pronunciation, content or vocabulary.",
           "{rubric}\n\nBelow is an automatic transcript of a 45-60 second spoken answer by an English learner. "
           "Punctuation was added by the speech recogniser and may be imperfect; judge the grammar of what was said.\n\n"
           "Transcript:\n\"\"\"{text}\"\"\"\n\nWhat grammar score (1-5) does this speaker deserve? Answer with a single digit only."),
}


def ask(model, system, user, url="http://localhost:11434/api/chat"):
    body = {"model": model, "stream": False, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"num_predict": 1, "temperature": 0, "num_ctx": 2048}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    top = out["logprobs"][0]["top_logprobs"]
    p = np.zeros(5)
    for t in top:
        tok = t["token"].strip()
        if tok in {"1", "2", "3", "4", "5"}:
            p[int(tok) - 1] += np.exp(t["logprob"])
    mass = p.sum()
    p = p / mass if mass > 0 else np.full(5, 0.2)
    return {"judge_exp": float((p * np.arange(1, 6)).sum()), "judge_entropy": float(-(p * np.log(p + 1e-12)).sum()),
            "judge_mass": float(mass), **{f"judge_p{k + 1}": float(p[k]) for k in range(5)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = f"{args.model.replace(':', '_').replace('/', '_')}_{args.prompt}"
    out_path = os.path.join(ROOT, "cache", f"judge_{tag}.csv")

    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    tr = pd.read_csv(os.path.join(ROOT, "cache", "transcripts.csv"), keep_default_na=False)
    meta = meta.merge(tr, on=["split", "filename"], how="left")
    done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=["split", "filename"])
    seen = set(zip(done.split, done.filename))
    todo = [r for r in meta.itertuples() if (r.split, r.filename) not in seen][:args.limit]
    system, user_t = PROMPTS[args.prompt]
    rows = []
    for i, r in enumerate(todo, 1):
        rows.append({"split": r.split, "filename": r.filename,
                     **ask(args.model, system, user_t.format(rubric=RUBRIC, text=clean(r.transcript or "")))})
        if i % 50 == 0 or i == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(out_path, index=False); rows = []
            print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
