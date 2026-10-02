"""Fine-tune WavLM end-to-end on the grammar labels with 5-fold CV (training clips only).

For each fold: train on 4/5 of the rubric-scored training clips, predict the held-out 1/5 (out-of-fold) and the
test set. Saves cache/ft_<tag>.npz with oof (n_train,) and test (n_test,) = mean over fold models.
Architecture: WavLM (CNN front-end frozen) -> learnable softmax-weighted sum of hidden layers -> mean pooling
-> MLP -> 5 * sigmoid, so outputs are bounded to [0, 5]. Loss: MSE.
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel

from asr_transcribe import DATA, ROOT, SR, load_audio


class Scorer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, layerdrop=0.0)   # fixed layer count for the weighted sum
        self.enc.feature_extractor._freeze_parameters()
        self.enc.gradient_checkpointing_enable()     # trade compute for memory on a 6 GB GPU
        with torch.no_grad():                       # count the hidden states this transformers version returns
            n_layers = len(self.enc(torch.zeros(1, SR), output_hidden_states=True).hidden_states)
        dim = self.enc.config.hidden_size
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x):
        hs = torch.stack(self.enc(x, output_hidden_states=True).hidden_states)      # L, B, T, D
        h = (torch.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0).mean(1)
        return 5 * torch.sigmoid(self.head(h).squeeze(-1))


def normalise(y):
    return (y - y.mean()) / (y.std() + 1e-7)


@torch.no_grad()
def predict(model, clips, win=20 * SR):
    model.eval()
    out = []
    for y in clips:
        y = y.astype(np.float32)
        segs = [y[a:a + win] for a in range(0, len(y), win) if a == 0 or len(y) - a >= 5 * SR]
        with torch.autocast("cuda", dtype=torch.float16):
            p = [model(torch.from_numpy(normalise(s))[None].cuda()).float().item() for s in segs]
        out.append(float(np.average(p, weights=[len(s) for s in segs])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--tag", default="wavlm_base")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=float, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    meta = pd.concat([pd.read_csv(os.path.join(DATA, f"{s}.csv")).assign(split=s) for s in ("train", "test")], ignore_index=True)
    tr_mask = ((meta.split == "train") & (meta.label > 0)).values
    te_mask = (meta.split == "test").values
    print("loading audio...", flush=True)
    audio = [load_audio(os.path.join(DATA, s, f)).astype(np.float16) for s, f in zip(meta.split, meta.filename)]
    X_tr = [a for a, m in zip(audio, tr_mask) if m]
    X_te = [a for a, m in zip(audio, te_mask) if m]
    y = meta.label[tr_mask].values.astype(np.float32)
    bins = np.digitize(y, [2.25, 2.75, 3.25, 3.75, 4.25, 4.75])
    crop = int(args.crop * SR)

    oof, test = np.zeros(len(y)), np.zeros(len(X_te))
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(y, bins)):
        t0 = time.time()
        model = Scorer(args.model).cuda()
        enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": 1e-3}], weight_decay=0.01)
        steps = args.epochs * math.ceil(len(tr) / args.bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / (0.1 * steps)) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
        scaler = torch.amp.GradScaler()
        for ep in range(args.epochs):
            model.train()
            perm = np.random.permutation(tr)
            losses = []
            for b in range(0, len(perm), args.bs):
                idx = perm[b:b + args.bs]
                batch = []
                for i in idx:
                    a = X_tr[i].astype(np.float32)
                    if len(a) > crop:
                        s = np.random.randint(0, len(a) - crop); a = a[s:s + crop]
                    else:
                        a = np.pad(a, (0, crop - len(a)))
                    batch.append(normalise(a * np.random.uniform(0.7, 1.0)))       # light gain augmentation
                x = torch.from_numpy(np.stack(batch)).cuda()
                t = torch.from_numpy(y[idx]).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = ((model(x).float() - t) ** 2).mean()
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
                losses.append(loss.item())
            print(f"fold {fold} ep {ep} train_mse {np.mean(losses):.3f} {time.time() - t0:.0f}s", flush=True)
            if ep == args.epochs - 1:
                pv = predict(model, [X_tr[i] for i in va])
                print(f"fold {fold} ep {ep} train_mse {np.mean(losses):.3f} val_rmse {np.sqrt(((pv - y[va]) ** 2).mean()):.4f}", flush=True)
        oof[va] = predict(model, [X_tr[i] for i in va])
        test += predict(model, X_te) / 5
        print(f"fold {fold} done in {time.time() - t0:.0f}s", flush=True)
        del model, opt; torch.cuda.empty_cache()
    rmse = np.sqrt(((oof - y) ** 2).mean())
    print(f"OOF RMSE {rmse:.4f}", flush=True)
    np.savez(os.path.join(ROOT, "cache", f"ft_{args.tag}.npz"), oof=oof, test=test)


if __name__ == "__main__":
    main()
