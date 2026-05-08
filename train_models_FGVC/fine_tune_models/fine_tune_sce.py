import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

BASE_DIR  = "/data/user"
FEAT_DIR  = os.path.join(BASE_DIR, "dinov2_features")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints/fgvc/fine_tune_sce_0.7_smooth_adj")
TB_DIR    = os.path.join(BASE_DIR, "dinov2_runs_fgvc/fine_tune_sce_0.7_smooth_adj")
FEAT_DIM  = 1024
N_LEAVES  = 102
BATCH     = 1024
EPOCHS    = 2000
LR        = 1e-1 / (10/7)
MOMENTUM  = 0.9
WD        = 0.0
WORKERS   = 8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="1")
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

S = torch.from_numpy(
    pd.read_csv("fgvc_sibling_matrix.csv", index_col=0)
    .values[:N_LEAVES, :N_LEAVES].astype(np.float32)
).to(device)


def load(split):
    d = torch.load(os.path.join(FEAT_DIR, f"fgvc_{split}_dinov2.pt"))
    return d["features"], d["fine_labels"]


tf, tl = load("trainval")
vf, vl = load("test")

train_loader = DataLoader(TensorDataset(tf, tl), batch_size=BATCH,
                          shuffle=True,  num_workers=WORKERS, pin_memory=True)
test_loader  = DataLoader(TensorDataset(vf, vl), batch_size=BATCH,
                          shuffle=False, num_workers=WORKERS, pin_memory=True)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


def metrics(logits, targets):
    pred1 = logits.argmax(1)
    c1    = (pred1 == targets).sum().item()
    _, top5 = logits.topk(5, dim=1)
    c5    = top5.eq(targets.unsqueeze(1)).sum().item()
    cs    = S[targets.cpu(), pred1.cpu()].sum().item()
    return c1, c5, cs


torch.manual_seed(42)
head      = nn.Linear(FEAT_DIM, N_LEAVES).to(device)
optimizer = torch.optim.SGD(head.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-2)

Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(TB_DIR)

best_top1, best_top5, best_val_loss = -1.0, -1.0, float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    head.train()
    t0 = time.time()
    rl, ns, nb, c1t, c5t, cst = 0.0, 0, 0, 0, 0, 0
    for feats, targets in train_loader:
        feats, targets = feats.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = head(feats)
        loss   = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        B = len(feats); ns += B; nb += 1; rl += loss.item()
        with torch.no_grad():
            c1, c5, cs = metrics(logits, targets)
        c1t += c1; c5t += c5; cst += cs

    tl_loss = rl / nb
    train_m = {"train_loss": tl_loss, "train_top1": c1t/ns,
               "train_top5": c5t/ns,  "train_sibling_acc": cst/ns}
    writer.add_scalar("Train/loss",        tl_loss,        epoch)
    writer.add_scalar("Train/top1",        100*c1t/ns,     epoch)
    writer.add_scalar("Train/top5",        100*c5t/ns,     epoch)
    writer.add_scalar("Train/sibling_acc", 100*cst/ns,     epoch)
    print(f"  [train] loss={tl_loss:.4f}  top1={100*c1t/ns:.1f}%  "
          f"top5={100*c5t/ns:.1f}%  sibling={100*cst/ns:.1f}%  ({time.time()-t0:.1f}s)")

    scheduler.step()

    head.eval()
    rl, ns, nb, c1t, c5t, cst = 0.0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for feats, targets in test_loader:
            feats, targets = feats.to(device), targets.to(device)
            logits = head(feats)
            rl += loss_fn(logits, targets).item()
            B = len(feats); ns += B; nb += 1
            c1, c5, cs = metrics(logits, targets)
            c1t += c1; c5t += c5; cst += cs

    vl_loss = rl / nb
    val_m = {"val_loss": vl_loss, "val_top1": c1t/ns,
             "val_top5": c5t/ns,  "val_sibling_acc": cst/ns}
    writer.add_scalar("Val/loss",        vl_loss,    epoch)
    writer.add_scalar("Val/top1",        100*c1t/ns, epoch)
    writer.add_scalar("Val/top5",        100*c5t/ns, epoch)
    writer.add_scalar("Val/sibling_acc", 100*cst/ns, epoch)
    print(f"  [val]   loss={vl_loss:.4f}  top1={100*c1t/ns:.1f}%  "
          f"top5={100*c5t/ns:.1f}%  sibling={100*cst/ns:.1f}%")

    ckpt = {
        "epoch": epoch, "model": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": {**train_m, **val_m},
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, "checkpoint.pth"))
    if val_m["val_top1"] > best_top1:
        best_top1 = val_m["val_top1"]
        torch.save(ckpt, os.path.join(CKPT_DIR, "checkpoint_best_top1.pth"))
    if val_m["val_top5"] > best_top5:
        best_top5 = val_m["val_top5"]
        torch.save(ckpt, os.path.join(CKPT_DIR, "checkpoint_best_top5.pth"))
    if val_m["val_loss"] < best_val_loss:
        best_val_loss = val_m["val_loss"]
        torch.save(ckpt, os.path.join(CKPT_DIR, "checkpoint_best_loss.pth"))

print(f"\nDone. Best top1: {100*best_top1:.2f}%  top5: {100*best_top5:.2f}%")
writer.close()