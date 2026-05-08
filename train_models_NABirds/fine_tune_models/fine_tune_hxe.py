import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",   default="6")
parser.add_argument("--alpha", type=float, default=0.5, choices=[0.2, 0.5, 0])
args   = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/data/user"
FEAT_DIR = os.path.join(BASE_DIR, "dinov2_features")
ALPHA_TAG = str(args.alpha).replace(".", "")
CKPT_DIR  = os.path.join(BASE_DIR, f"checkpoints/NABirds/fine_tune_hxe_{ALPHA_TAG}3")
TB_DIR    = os.path.join(BASE_DIR, f"dinov2_runs_NABirds/fine_tune_hxe_{ALPHA_TAG}3")

FEAT_DIM = 1024
BATCH    = 1024
EPOCHS   = 2000
LR       = 1e-1
MOMENTUM = 0.9
WD       = 0.0
WORKERS  = 8

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

child_leaf_masks  = torch.load("child_leaf_masks.pt").float().to(device)
parent_leaf_masks = torch.load("parent_leaf_masks.pt").float().to(device)

N_LEAVES = child_leaf_masks.shape[0]
E        = child_leaf_masks.shape[1]

depth_matrix = torch.from_numpy(
    pd.read_csv("depth_matrix.csv", index_col=0).values.astype(np.float32)
).to(device)

edge_weights = torch.exp(-args.alpha * depth_matrix)

print(f"N_LEAVES={N_LEAVES}, E={E}")
print(f"child_leaf_masks: {child_leaf_masks.shape}, "
      f"parent_leaf_masks: {parent_leaf_masks.shape}, "
      f"edge_weights: {edge_weights.shape}")


leaf_df   = pd.read_csv("leaf_index.csv", index_col=0)
leaf_list = leaf_df["original_node_id"].tolist()

N_TOTAL     = 1011
orig_to_leaf = torch.full((N_TOTAL,), -1, dtype=torch.long)
for leaf_idx, orig_id in enumerate(leaf_list):
    orig_to_leaf[orig_id] = leaf_idx

# ---------------------------------------------------------------------------
# Sibling matrix: original is [1011, 1011] indexed by orig node IDs.
# Reindex to [N_LEAVES, N_LEAVES] so it aligns with leaf-space predictions.
# ---------------------------------------------------------------------------
S_full = torch.from_numpy(
    pd.read_csv("sibling_matrix.csv", index_col=0).values.astype(np.float32)
)                                                             # [1011, 1011]
S = S_full[leaf_list][:, leaf_list].to(device)              # [N_LEAVES, N_LEAVES]


class HXELoss(nn.Module):
    def __init__(self, child_masks, parent_masks, weights):
        super().__init__()
        self.register_buffer("child_masks",  child_masks)
        self.register_buffer("parent_masks", parent_masks)
        self.register_buffer("weights",      weights)

    def forward(self, logits, targets):
        # logits:  [B, N_LEAVES]   (softmax over leaves only)
        # targets: [B]             (leaf indices 0..N_LEAVES-1)
        probs = F.softmax(logits, dim=1)                              # [B, N_LEAVES]
        c_masks = self.child_masks[targets]                           # [B, E, N_LEAVES]
        p_masks = self.parent_masks[targets]                          # [B, E, N_LEAVES]
        w       = self.weights[targets]                               # [B, E]

        child_mass  = (c_masks * probs.unsqueeze(1)).sum(dim=-1)      # [B, E]
        parent_mass = (p_masks * probs.unsqueeze(1)).sum(dim=-1)      # [B, E]

        cond      = child_mass / parent_mass.clamp_min(1e-12)
        edge_loss = -torch.log(cond.clamp_min(1e-12))                 # [B, E]

        return (w * edge_loss).sum(dim=1).mean()


criterion = HXELoss(child_leaf_masks, parent_leaf_masks, edge_weights)


def verify_mass_conservation(loss_module, n_leaves, dev):
    uniform     = torch.full((1, n_leaves), 1.0 / n_leaves, device=dev)
    all_targets = torch.arange(n_leaves, device=dev)

    c_masks = loss_module.child_masks[all_targets]   # [N_LEAVES, E, N_LEAVES]
    p_masks = loss_module.parent_masks[all_targets]  # [N_LEAVES, E, N_LEAVES]

    child_mass  = (c_masks * uniform).sum(dim=-1)    # [N_LEAVES, E]
    parent_mass = (p_masks * uniform).sum(dim=-1)    # [N_LEAVES, E]

    if not (child_mass <= parent_mass + 1e-6).all():
        bad = (child_mass > parent_mass + 1e-6).nonzero(as_tuple=False)
        raise RuntimeError(f"Mass conservation violated (child > parent) at {bad.tolist()}")

    if not (parent_mass <= 1.0 + 1e-6).all():
        bad = (parent_mass > 1.0 + 1e-6).nonzero(as_tuple=False)
        raise RuntimeError(f"Mass conservation violated (parent > 1) at {bad.tolist()}")

    print("Mass conservation check passed: child_mass <= parent_mass <= 1 for all edges.")

verify_mass_conservation(criterion, N_LEAVES, device)


def load(split):
    d = torch.load(os.path.join(FEAT_DIR, f"nabirds_{split}_dinov2.pt"))
    features    = d["features"]
    # Remap original node IDs -> contiguous leaf indices
    fine_labels = orig_to_leaf[d["fine_labels"]]
    assert (fine_labels >= 0).all(), \
        "Some labels in the dataset are not leaf nodes — check hierarchy or label mapping"
    return features, fine_labels

tf, tl = load("train")
vf, vl = load("test")
train_loader = DataLoader(TensorDataset(tf, tl), batch_size=BATCH,
                          shuffle=True,  num_workers=WORKERS, pin_memory=True)
test_loader  = DataLoader(TensorDataset(vf, vl), batch_size=BATCH,
                          shuffle=False, num_workers=WORKERS, pin_memory=True)


def metrics(logits, targets):
    # Both logits and targets are in leaf-index space (0..N_LEAVES-1)
    pred1    = logits.argmax(1)
    c1       = (pred1 == targets).sum().item()
    _, top5  = logits.topk(5, dim=1)
    c5       = top5.eq(targets.unsqueeze(1)).sum().item()
    cs       = S[targets.cpu(), pred1.cpu()].sum().item()
    return c1, c5, cs

torch.manual_seed(42)
head      = nn.Linear(FEAT_DIM, N_LEAVES).to(device)   # outputs N_LEAVES logits only
optimizer = torch.optim.SGD(head.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-2)

Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(TB_DIR)

best_top1, best_top5, best_val_loss = -1.0, -1.0, float("inf")


for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}  [alpha={args.alpha}]")

    head.train()
    t0 = time.time()
    rl, ns, nb, c1t, c5t, cst = 0.0, 0, 0, 0, 0, 0
    for feats, targets in train_loader:
        feats, targets = feats.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = head(feats)
        loss   = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        B = len(feats); ns += B; nb += 1; rl += loss.item()
        with torch.no_grad():
            c1, c5, cs = metrics(logits, targets)
        c1t += c1; c5t += c5; cst += cs

    tl_loss  = rl / nb
    train_m  = {"train_loss": tl_loss, "train_top1": c1t/ns,
                "train_top5": c5t/ns,  "train_sibling_acc": cst/ns}
    writer.add_scalar("Train/loss",        tl_loss,       epoch)
    writer.add_scalar("Train/top1",        100*c1t/ns,    epoch)
    writer.add_scalar("Train/top5",        100*c5t/ns,    epoch)
    writer.add_scalar("Train/sibling_acc", 100*cst/ns,    epoch)
    print(f"  [train] loss={tl_loss:.4f}  top1={100*c1t/ns:.1f}%  "
          f"top5={100*c5t/ns:.1f}%  sibling={100*cst/ns:.1f}%  ({time.time()-t0:.1f}s)")

    scheduler.step()

    head.eval()
    rl, ns, nb, c1t, c5t, cst = 0.0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for feats, targets in test_loader:
            feats, targets = feats.to(device), targets.to(device)
            logits = head(feats)
            rl += criterion(logits, targets).item()
            B = len(feats); ns += B; nb += 1
            c1, c5, cs = metrics(logits, targets)
            c1t += c1; c5t += c5; cst += cs

    vl_loss = rl / nb
    val_m   = {"val_loss": vl_loss, "val_top1": c1t/ns,
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
        "alpha": args.alpha,
        "n_leaves": N_LEAVES,
        "leaf_list": leaf_list,
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