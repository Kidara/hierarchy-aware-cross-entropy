"""
compare_fgvc.py
Produces a Top-1 and Top-5 accuracy delta matrix (HCE - SCE) for FGVC Aircraft.

Comparison pairs (6 per architecture):
  hce_0.2     vs sce_0.2_adj      |  hce_0.2_adj vs sce
  hce_0.5     vs sce_0.5_adj      |  hce_0.5_adj vs sce
  hce_0.7     vs sce_0.7_adj      |  hce_0.7_adj vs sce
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from swinT_hce import test_loader

BASE_DIR  = "/data/user/checkpoints/fgvc"
CKPT_FILE = "checkpoint.pth"

NUM_SCE = 102
NUM_HCE = 201

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# sentinel for a checkpoint that doesn't exist yet
MISSING = "MISSING"

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
        "hce_0.2_adj": "convnext_hce_0.2_no_smooth",
        "hce_0.5":     "convnext_hce_0.5_no_smooth",
        "hce_0.5_adj": "convnext_hce_0.5_no_smooth_adj",
        "hce_0.7":     "convnext_hce_0.7_no_smooth",
        "hce_0.7_adj": "convnext_hce_0.7_no_smooth_adj",
        "sce":         "convnext_sce_no_smooth",
        "sce_0.2_adj": "convnext_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "convnext_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "convnext_sce_0.7_no_smooth_adj",
    }),
    ("swint", make_swint, {
        "hce_0.2":     "swint_hce_0.2_no_smooth",
        "hce_0.2_adj": "swint_hce_0.2_no_smooth_adj",
        "hce_0.5":     "swint_hce_0.5_no_smooth",
        "hce_0.5_adj": "swint_hce_0.5_no_smooth_adj",
        "hce_0.7":     "swint_hce_0.7_no_smooth",
        "hce_0.7_adj": "swint_hce_0.7_no_smooth_adj",
        "sce":         "swint_sce_no_smooth",
        "sce_0.2_adj": "swint_sce_0.2_no_smooth_adj",
        "sce_0.5_adj": "swint_sce_0.5_no_smooth_adj",
        "sce_0.7_adj": "swint_sce_0.7_no_smooth_adj",
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
    data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model"])
    model.eval()
    return model

@torch.no_grad()
def evaluate_single(model, k=5):
    """Returns (top1, top5) for a single model."""
    total = correct1 = correct5 = 0
    for X, labels in test_loader:
        X, labels = X.to(device), labels.to(device)
        logits = model(X)
        correct1 += (logits.argmax(1) == labels).sum().item()
        topk = torch.topk(logits, k=k, dim=1).indices
        correct5 += sum(labels[i] in topk[i] for i in range(len(labels)))
        total += len(labels)
    return 100 * correct1 / total, 100 * correct5 / total


def best_across(make_fn, num_classes, folders):
    """Evaluate all folders, return the best top1 and top5 seen."""
    best_top1, best_top5 = -1.0, -1.0
    best_top1_folder, best_top5_folder = None, None

    for folder in folders:
        model = load_model(make_fn, num_classes, folder)
        if model is None:
            print(f"    MISSING: {folder}")
            continue
        top1, top5 = evaluate_single(model)
        print(f"    {folder}: top1={top1:.2f}%  top5={top5:.2f}%")
        if top1 > best_top1:
            best_top1, best_top1_folder = top1, folder
        if top5 > best_top5:
            best_top5, best_top5_folder = top5, folder

    return best_top1, best_top5, best_top1_folder, best_top5_folder


# ── Main loop ────────────────────────────────────────────────────────────────

arch_names  = [cfg[0] for cfg in ARCH_CONFIGS]
comp_labels = [c[0] for c in COMPARISONS]

delta_top1 = np.full((len(arch_names), len(COMPARISONS)), np.nan)
delta_top5 = np.full((len(arch_names), len(COMPARISONS)), np.nan)

# Cache per (arch, folder) so the same checkpoint isn't evaluated twice
# (e.g. `sce` appears in both "λ=0.2 adj" and "λ=0.5 adj" comparisons).
eval_cache = {}

def eval_folder(arch_name, make_fn, num_classes, folder):
    key = (arch_name, folder)
    if key in eval_cache:
        return eval_cache[key]
    model = load_model(make_fn, num_classes, folder)
    if model is None:
        print(f"    MISSING: {folder}")
        eval_cache[key] = None
        return None
    top1, top5 = evaluate_single(model)
    print(f"    {folder}: top1={top1:.2f}%  top5={top5:.2f}%")
    eval_cache[key] = (top1, top5)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return eval_cache[key]

for i, (arch_name, make_fn, folders) in enumerate(ARCH_CONFIGS):
    print(f"\n{'='*60}\n{arch_name}\n{'='*60}")
    for j, (label, hce_key, sce_key) in enumerate(COMPARISONS):
        print(f"  [{label}]  HCE={hce_key}  vs  SCE={sce_key}")
        hce_folder = folders.get(hce_key, MISSING)
        sce_folder = folders.get(sce_key, MISSING)
        hce_res = eval_folder(arch_name, make_fn, NUM_HCE, hce_folder)
        sce_res = eval_folder(arch_name, make_fn, NUM_SCE, sce_folder)
        if hce_res is not None and sce_res is not None:
            delta_top1[i, j] = hce_res[0] - sce_res[0]
            delta_top5[i, j] = hce_res[1] - sce_res[1]

# ── Text summary ──────────────────────────────────────────────────────────────

def print_matrix(mat, title):
    print(f"\n{title}")
    print(f"{'Arch':<12} " + " ".join(f"{l:>12}" for l in comp_labels))
    for i, name in enumerate(arch_names):
        cells = []
        for j in range(len(comp_labels)):
            v = mat[i, j]
            cells.append(f"{'N/A':>12}" if np.isnan(v) else f"{v:>+12.2f}")
        print(f"{name:<12} " + " ".join(cells))

print_matrix(delta_top1, "Δ Top-1 (HCE − SCE, percentage points)")
print_matrix(delta_top5, "Δ Top-5 (HCE − SCE, percentage points)")

# ── Heatmaps ──────────────────────────────────────────────────────────────────

def plot_matrix(mat, title, filename):
    finite = mat[np.isfinite(mat)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(arch_names)))
    ax.set_yticklabels(arch_names)

    for i in range(len(arch_names)):
        for j in range(len(comp_labels)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center",
                        color="black", fontsize=9)
            else:
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color=color, fontsize=10)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Δ accuracy (pp)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")

plot_matrix(delta_top1, "Δ Top-1 accuracy  (HCE − SCE)", "matrix_accuracy_top1.png")
plot_matrix(delta_top5, "Δ Top-5 accuracy  (HCE − SCE)", "matrix_accuracy_top5.png")