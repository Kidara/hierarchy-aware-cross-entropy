import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from convnext_hce import test_loader


BASE_DIR = "/data/user/checkpoints/NA_Birds"
NAB_DIR  = "/data/user/nabirds"
CKPT_FILE = "checkpoint.pth"

NUM_SCE   = 555
NUM_HCE   = 1011
MISSING   = "MISSING"

device = "cuda:6" if torch.cuda.is_available() else "cpu"

import os
import torch

BASE_DIR  = "/data/user/checkpoints/NA_Birds"
CKPT_FILE = "checkpoint.pth"

NUM_SCE = 555
NUM_HCE = 1011

raw_labels = {}
with open(os.path.join(NAB_DIR, "image_class_labels.txt")) as f:
    for line in f:
        img_id, label = line.strip().split()
        raw_labels[img_id] = int(label)

all_tax_ids     = sorted(set(raw_labels.values()))
taxonomy_to_seq = {tax_id: i for i, tax_id in enumerate(all_tax_ids)}
label_map       = torch.full((NUM_HCE,), -1, dtype=torch.long, device=device)
for node_id, seq_idx in taxonomy_to_seq.items():
    label_map[node_id] = seq_idx

import os

BASE_DIR  = "/data/user/checkpoints/NA_Birds"
CKPT_FILE = "checkpoint.pth"

def make_resnet18(n):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def make_resnet34(n):
    m = models.resnet34(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def make_resnet50(n):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def make_convnext(n):
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    return m

def make_swint(n):
    m = models.swin_t(weights=None)
    m.head = nn.Linear(m.head.in_features, n)
    return m

def make_vit(n):
    m = models.vit_b_16(weights=None, dropout=0.1)
    m.heads = nn.Linear(m.heads.head.in_features, n)
    return m



ARCH_CONFIGS = [
    ("resnet18", make_resnet18, {
        "hce_0.2":     "resnet18_hce_0.2_no_smooth",
        "hce_0.2_adj": "resnet18_hce_0.2_no_smooth_adj",
        "hce_0.5":     "resnet18_hce_0.5_no_smooth",
        "hce_0.5_adj": "resnet18_hce_0.5_no_smooth_adj",
        "hce_0.7":     "resnet18_hce_0.7_no_smooth",
        "hce_0.7_adj": "resnet18_hce_0.7_no_smooth_adj",
        "sce":         "resnet18_sce_no_smooth",                         
        "sce_0.2_adj": "resnet18_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "resnet18_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "resnet18_sce_0.7_no_smooth_adj",
    }),
    ("resnet34", make_resnet34, {
        "hce_0.2":     "resnet34_hce_0.2_no_smooth",
        "hce_0.2_adj": "resnet34_hce_0.2_no_smooth_adj",
        "hce_0.5":     "resnet34_hce_0.5_no_smooth",
        "hce_0.5_adj": "resnet34_hce_0.5_no_smooth_adj",
        "hce_0.7":     "resnet34_hce_0.7_no_smooth",
        "hce_0.7_adj": "resnet34_hce_0.7_no_smooth_adj",
        "sce":         "resnet34_sce_no_smooth",
        "sce_0.2_adj": "resnet34_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "resnet34_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "resnet34_sce_0.7_no_smooth_adj",
    }),
    ("resnet50", make_resnet50, {
        "hce_0.2":     "resnet50_hce_0.2_no_smooth",
        "hce_0.2_adj": "resnet50_hce_0.2_no_smooth_adj",
        "hce_0.5":     "resnet50_hce_0.5_no_smooth",
        "hce_0.5_adj": "resnet50_hce_0.5_no_smooth_adj",
        "hce_0.7":     "resnet50_hce_0.7_no_smooth",
        "hce_0.7_adj": "resnet50_hce_0.7_no_smooth_adj",
        "sce":         "resnet50_sce_no_smooth",
        "sce_0.2_adj": "resnet50_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "resnet50_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "resnet50_sce_0.7_no_smooth_adj",
    }),
    ("convnext", make_convnext, {
        "hce_0.2":     "convnext_hce_0.2_no_smooth",
        "hce_0.2_adj": "convnext_hce_0.2_no_smooth_adj",
        "hce_0.5":     "convnext_hce_0.5_no_smooth",
        "hce_0.5_adj": "convnext_hce_0.5_no_smooth_adj",
        "hce_0.7":     "convnext_hce_0.7_no_smooth",
        "hce_0.7_adj": "convnext_hce_0.7_no_smooth_adj",
        "sce":         "convnext_sce_no_smooth",
        "sce_0.2_adj": "convnext_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "convnext_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "convnext_sce_0.7_no_smooth_adj",
    }),
    ("swinT", make_swint, {
        "hce_0.2":     "swinT_hce_0.2_no_smooth",
        "hce_0.2_adj": "swinT_hce_0.2_no_smooth_adj",
        "hce_0.5":     "swinT_hce_0.5_no_smooth",
        "hce_0.5_adj": "swinT_hce_0.5_no_smooth_adj",
        "hce_0.7":     "swinT_hce_0.7_no_smooth",
        "hce_0.7_adj": "swinT_hce_0.7_no_smooth_adj",
        "sce":         "swinT_sce_no_smooth",
        "sce_0.2_adj": "swinT_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "swinT_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "swinT_sce_0.7_no_smooth_adj",
    }),
    ("vit", make_vit, {
        "hce_0.2":     "vit_hce_0.2_no_smooth",
        "hce_0.2_adj": "vit_hce_0.2_no_smooth_adj",
        "hce_0.5":     "vit_hce_0.5_no_smooth",
        "hce_0.5_adj": "vit_hce_0.5_no_smooth_adj",
        "hce_0.7":     "vit_hce_0.7_no_smooth",
        "hce_0.7_adj": "vit_hce_0.7_no_smooth_adj",
        "sce":         "vit_sce_no_smooth",
        "sce_0.2_adj": "vit_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "vit_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "vit_sce_0.7_no_smooth_adj",
    }),
]

COMPARISONS = [
    ("λ=0.2",     "hce_0.2",     "sce_0.2_adj"),
    ("λ=0.2 adj", "hce_0.2_adj", "sce"),
    ("λ=0.5",     "hce_0.5",     "sce_0.5_adj"),
    ("λ=0.5 adj", "hce_0.5_adj", "sce"),
    ("λ=0.7",     "hce_0.7",     "sce_0.7_adj"),
    ("λ=0.7 adj", "hce_0.7_adj", "sce"),
]


def load_model(make_fn, num_classes, folder):
    if folder == MISSING:
        return None
    path = os.path.join(BASE_DIR, folder, CKPT_FILE)
    model = make_fn(num_classes).to(device)
    data  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model"])
    model.eval()
    return model

@torch.no_grad()
def evaluate(model_hce, model_sce, k=5):
    """Returns (top1_hce, top5_hce, top1_sce, top5_sce). None if either model missing."""
    if model_hce is None or model_sce is None:
        return None

    total = correct1_hce = correct5_hce = correct1_sce = correct5_sce = 0

    for X, labels in test_loader:
        X, labels = X.to(device), labels.to(device)
        labels_sce = label_map[labels]

        logits_hce = model_hce(X)
        pred1_hce  = logits_hce.argmax(1)           # raw taxonomy index
        topk_hce   = torch.topk(logits_hce, k=k, dim=1).indices

        logits_sce = model_sce(X)
        pred1_sce  = logits_sce.argmax(1)            # sequential index
        topk_sce   = torch.topk(logits_sce, k=k, dim=1).indices

        correct1_hce += (pred1_hce == labels).sum().item()
        correct1_sce += (pred1_sce == labels_sce).sum().item()

        for i in range(len(labels)):
            if labels[i]     in topk_hce[i]: correct5_hce += 1
            if labels_sce[i] in topk_sce[i]: correct5_sce += 1

        total += len(labels)

    return (100 * correct1_hce / total, 100 * correct5_hce / total,
            100 * correct1_sce  / total, 100 * correct5_sce  / total)


arch_names = [cfg[0] for cfg in ARCH_CONFIGS]
col_labels = [c[0] for c in COMPARISONS]

delta1_matrix = np.full((len(ARCH_CONFIGS), len(COMPARISONS)), np.nan)
delta5_matrix = np.full((len(ARCH_CONFIGS), len(COMPARISONS)), np.nan)

# for row_idx, (arch_name, make_fn, folders) in enumerate(ARCH_CONFIGS):
#     print(f"\n{'='*60}\n{arch_name}\n{'='*60}")

#     sce_cache = {}
#     for key in ("sce", "sce_0.2_adj", "sce_0.5_adj", "sce_0.7_adj"):
#         sce_cache[key] = load_model(make_fn, NUM_SCE, folders[key])

#     for col_idx, (label, hce_key, sce_key) in enumerate(COMPARISONS):
#         model_hce = load_model(make_fn, NUM_HCE, folders[hce_key])
#         model_sce = sce_cache[sce_key]

#         result = evaluate(model_hce, model_sce)
#         if result is None:
#             print(f"  {label:12s}: MISSING")
#         else:
#             t1h, t5h, t1s, t5s = result
#             d1, d5 = t1h - t1s, t5h - t5s
#             delta1_matrix[row_idx, col_idx] = d1
#             delta5_matrix[row_idx, col_idx] = d5
#             print(f"  {label:12s}: Top-1 HCE {t1h:.2f}% vs SCE {t1s:.2f}% (Δ{d1:+.2f}%)  "
#                   f"Top-5 HCE {t5h:.2f}% vs SCE {t5s:.2f}% (Δ{d5:+.2f}%)")


def plot_matrix(matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    vmax = np.nanmax(np.abs(matrix)) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_labels)));  ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(arch_names)));  ax.set_yticklabels(arch_names)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Δ Accuracy (HCE − SCE, %)")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            txt = f"{val:+.1f}" if not np.isnan(val) else "N/A"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < vmax * 0.6 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

OUT = "/data/user/checkpoints/NA_Birds"
plot_matrix(delta1_matrix, "NA Birds — Top-1 Δ (HCE − SCE)", os.path.join(OUT, "matrix_nabirds_top1.png"))
plot_matrix(delta5_matrix, "NA Birds — Top-5 Δ (HCE − SCE)", os.path.join(OUT, "matrix_nabirds_top5.png"))

SMOOTH_CKPTS = [
    ("resnet18", make_resnet18, "resnet18_sce_smooth"),
    ("resnet34", make_resnet34, "resnet34_sce_smooth"),
    ("resnet50", make_resnet50, "resnet50_sce_smooth"),
    ("convnext", make_convnext, "convnext_sce_smooth"),
    ("swinT",    make_swint,    "swinT_sce_smooth"),
    ("vit",      make_vit,      "vit_sce_smooth"),
]

def evaluate_sce(model, k=5):
    total = correct1 = correct5 = 0
    for X, labels in test_loader:
        X, labels = X.to(device), labels.to(device)
        labels_seq = label_map[labels]          # map taxonomy → sequential

        logits = model(X)
        pred1  = logits.argmax(1)
        topk   = torch.topk(logits, k=k, dim=1).indices

        correct1 += (pred1 == labels_seq).sum().item()
        for i in range(len(labels_seq)):
            if labels_seq[i] in topk[i]:
                correct5 += 1
        total += len(labels)

    return 100 * correct1 / total, 100 * correct5 / total

for arch_name, make_fn, folder in SMOOTH_CKPTS:
    path = os.path.join(BASE_DIR, folder, CKPT_FILE)
    if not os.path.exists(path):
        print(f"{folder:<30}  {'MISSING':>6}")
        continue

    model = make_fn(NUM_SCE).to(device)
    data  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model"])
    model.eval()

    top1, top5 = evaluate_sce(model)
    print(f"{folder:<30}  {top1:>6.2f}  {top5:>6.2f}")

    del model
    torch.cuda.empty_cache()

# ── HCE smooth checkpoints ──────────────────────────────────────────────────
SMOOTH_HCE_CKPTS = [
    ("resnet18", make_resnet18, "resnet18_hce_0.2_smooth_adj"),
    ("resnet34", make_resnet34, "resnet34_hce_0.2_smooth"),
    ("resnet50", make_resnet50, "resnet50_hce_0.2_smooth"),
    ("vit",      make_vit,      "vit_hce_0.5_smooth_adj"),
    ("convnext", make_convnext, "convnext_hce_0.2_smooth_adj"),
    ("swinT", make_swint, "swinT_hce_0.5_smooth_adj"),
]

@torch.no_grad()
def evaluate_hce(model, k=5):
    """Top-1 / Top-5 for an HCE model (raw taxonomy labels, NUM_HCE outputs)."""
    total = correct1 = correct5 = 0
    for X, labels in test_loader:
        X, labels = X.to(device), labels.to(device)   # labels = raw taxonomy ids

        logits = model(X)
        pred1  = logits.argmax(1)
        topk   = torch.topk(logits, k=k, dim=1).indices

        correct1 += (pred1 == labels).sum().item()
        for i in range(len(labels)):
            if labels[i] in topk[i]:
                correct5 += 1
        total += len(labels)

    return 100 * correct1 / total, 100 * correct5 / total

print(f"\n{'='*60}\nHCE Smooth Checkpoints\n{'='*60}")
print(f"{'Folder':<40}  {'Top-1':>6}  {'Top-5':>6}")
for arch_name, make_fn, folder in SMOOTH_HCE_CKPTS:
    path = os.path.join(BASE_DIR, folder, CKPT_FILE)
    if not os.path.exists(path):
        print(f"{folder:<40}  {'MISSING':>6}")
        continue

    model = make_fn(NUM_HCE).to(device)
    data  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model"])
    model.eval()

    top1, top5 = evaluate_hce(model)
    print(f"{folder:<40}  {top1:>6.2f}  {top5:>6.2f}")

    del model
    torch.cuda.empty_cache()