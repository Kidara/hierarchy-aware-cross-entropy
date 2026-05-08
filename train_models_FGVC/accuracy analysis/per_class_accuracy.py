"""
per_class_accuracy.py
Per-class accuracy at three hierarchy levels (variants, families, manufacturers)
for FGVC Aircraft. SCE in grey, HCE delta in green/red. Sorted by SCE desc.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

from swinT_hce import test_loader

BASE_DIR  = "/data/user/checkpoints/fgvc"
CKPT_FILE = "checkpoint_best_top1.pth"

# Hierarchy sizes
NUM_LEAVES = 100    # leaves with actual test data (test labels 0..99)
NUM_SCE    = 102    # SCE model's fc output dim (checkpoint has 2 extra dead classes)
NUM_HCE    = 201    # HCE model's fc output dim

device = "cuda:1" if torch.cuda.is_available() else "cpu"


# ── Load hierarchy + names ────────────────────────────────────────────────────

import re

def remove_clippath(svg_path):
    with open(svg_path, 'r') as f:
        content = f.read()
    # Remove clip-path attributes
    content = re.sub(r'\s*clip-path="url\(#[^"]+\)"', '', content)
    # Remove clipPath definitions
    content = re.sub(r'<clipPath[^>]*>.*?</clipPath>', '', content, flags=re.DOTALL)
    with open(svg_path, 'w') as f:
        f.write(content)

def load_child_to_parent(path="fgvc_hierarchy.txt"):
    mapping = {}
    with open(path) as f:
        for line in f:
            c, p = line.strip().split()
            mapping[int(c)] = int(p)
    return mapping

def load_names(path="fgvc_classes.txt"):
    names = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                names[int(parts[0])] = parts[1]
    print(f"Loaded {len(names)} names from {path}")
    return names

child_to_parent = load_child_to_parent()
class_names     = load_names()

# leaf (0..99) → family, family → manufacturer
leaf_to_family  = np.array([child_to_parent[l] for l in range(NUM_LEAVES)])
family_ids      = sorted(set(leaf_to_family.tolist()))
family_to_manuf = {f: child_to_parent[f] for f in family_ids}
manuf_ids       = sorted(set(family_to_manuf.values()))

NUM_FAMILIES = len(family_ids)
NUM_MANUF    = len(manuf_ids)
print(f"Leaves: {NUM_LEAVES}, Families: {NUM_FAMILIES}, Manufacturers: {NUM_MANUF}")

family_to_col  = {f: i for i, f in enumerate(family_ids)}
manuf_to_col   = {m: i for i, m in enumerate(manuf_ids)}

leaf_fam_col   = np.array([family_to_col[leaf_to_family[l]] for l in range(NUM_LEAVES)])
leaf_manuf_col = np.array([
    manuf_to_col[family_to_manuf[leaf_to_family[l]]] for l in range(NUM_LEAVES)
])

leaf_names   = [class_names[i] for i in range(NUM_LEAVES)]
family_names = [class_names[f] for f in family_ids]
manuf_names  = [class_names[m] for m in manuf_ids]


# ── Aggregation matrices ──────────────────────────────────────────────────────
# For each level, build a matrix A with shape (model_output_dim, num_classes_at_level).
# We compute: agg_probs = softmax(logits) @ A.
# SCE has 102 rows; only the first 100 are meaningful, the last 2 stay zero so
# any probability the SCE model assigns to phantom classes is dropped.

# LEAVES
A_sce_leaf = np.zeros((NUM_SCE, NUM_LEAVES), dtype=np.float32)
for l in range(NUM_LEAVES):
    A_sce_leaf[l, l] = 1

A_hce_leaf = np.zeros((NUM_HCE, NUM_LEAVES), dtype=np.float32)
for l in range(NUM_LEAVES):
    A_hce_leaf[l, l] = 1

# FAMILIES
A_sce_fam = np.zeros((NUM_SCE, NUM_FAMILIES), dtype=np.float32)
for l in range(NUM_LEAVES):
    A_sce_fam[l, leaf_fam_col[l]] = 1

A_hce_fam = np.zeros((NUM_HCE, NUM_FAMILIES), dtype=np.float32)
for l in range(NUM_LEAVES):
    A_hce_fam[l, leaf_fam_col[l]] = 1
for f in family_ids:
    A_hce_fam[f, family_to_col[f]] = 1

# MANUFACTURERS
A_sce_man = np.zeros((NUM_SCE, NUM_MANUF), dtype=np.float32)
for l in range(NUM_LEAVES):
    A_sce_man[l, leaf_manuf_col[l]] = 1

A_hce_man = np.zeros((NUM_HCE, NUM_MANUF), dtype=np.float32)
for l in range(NUM_LEAVES):
    A_hce_man[l, leaf_manuf_col[l]] = 1
for f in family_ids:
    A_hce_man[f, manuf_to_col[family_to_manuf[f]]] = 1
for m in manuf_ids:
    A_hce_man[m, manuf_to_col[m]] = 1

# Move to device
A_sce_leaf_t = torch.from_numpy(A_sce_leaf).to(device)
A_hce_leaf_t = torch.from_numpy(A_hce_leaf).to(device)
A_sce_fam_t  = torch.from_numpy(A_sce_fam).to(device)
A_hce_fam_t  = torch.from_numpy(A_hce_fam).to(device)
A_sce_man_t  = torch.from_numpy(A_sce_man).to(device)
A_hce_man_t  = torch.from_numpy(A_hce_man).to(device)

# Target column for each leaf label, at each level
leaf_target  = torch.arange(NUM_LEAVES, device=device)
fam_target   = torch.from_numpy(leaf_fam_col).to(device)
manuf_target = torch.from_numpy(leaf_manuf_col).to(device)


# ── Model loading ─────────────────────────────────────────────────────────────

def make_resnet50(n):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def load_model(make_fn, num_classes, folder):
    path = os.path.join(BASE_DIR, folder, CKPT_FILE)
    m = make_fn(num_classes).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


@torch.no_grad()
def per_class_acc(model, A, target_col, num_classes):
    """Aggregate probabilities through A, argmax, compare against target_col."""
    correct = np.zeros(num_classes)
    total   = np.zeros(num_classes)
    for X, labels in test_loader:
        X, labels = X.to(device), labels.to(device)
        probs  = torch.softmax(model(X), dim=1)
        agg    = probs @ A
        pred   = agg.argmax(1)
        target = target_col[labels]
        for c in range(num_classes):
            mask = target == c
            correct[c] += (pred[mask] == c).sum().item()
            total[c]   += mask.sum().item()
    return 100 * np.divide(correct, total, where=total > 0, out=np.zeros(num_classes))


# ── Evaluate ──────────────────────────────────────────────────────────────────

print("Loading SCE model...")
model_sce = load_model(make_resnet50, NUM_SCE, "resnet50_sce_no_smooth")
sce_leaf  = per_class_acc(model_sce, A_sce_leaf_t, leaf_target,  NUM_LEAVES)
sce_fam   = per_class_acc(model_sce, A_sce_fam_t,  fam_target,   NUM_FAMILIES)
sce_man   = per_class_acc(model_sce, A_sce_man_t,  manuf_target, NUM_MANUF)
del model_sce

print("Loading HCE model...")
model_hce = load_model(make_resnet50, NUM_HCE, "resnet50_hce_0.2_no_smooth_adj")
hce_leaf  = per_class_acc(model_hce, A_hce_leaf_t, leaf_target,  NUM_LEAVES)
hce_fam   = per_class_acc(model_hce, A_hce_fam_t,  fam_target,   NUM_FAMILIES)
hce_man   = per_class_acc(model_hce, A_hce_man_t,  manuf_target, NUM_MANUF)
del model_hce


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_level(sce_acc, hce_acc, names, title, save_path, label_fontsize=6):
    delta   = hce_acc - sce_acc
    order   = np.argsort(-sce_acc)
    sce_s   = sce_acc[order]
    delta_s = delta[order]
    names_s = [names[i] for i in order]
    n       = len(sce_s)
    x       = np.arange(n)
    width_inches = max(14, 0.22 * n)

    fig, ax = plt.subplots(figsize=(width_inches, 7))

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')

    # Light grey gridlines behind bars
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # No y-axis tick marks
    ax.tick_params(axis='y', left=False, labelsize=8)
    ax.tick_params(axis='x', pad=2)

    # Lighter, cooler grey for baseline
    ax.bar(x, sce_s, color='#b0b0b8', width=0.8, zorder=3)

    imp = delta_s > 0
    ax.bar(x[imp], delta_s[imp], bottom=sce_s[imp],
        color="green", alpha=0.8, width=0.8, zorder=4)
    wor = delta_s < 0
    ax.bar(x[wor], delta_s[wor], bottom=sce_s[wor],
        color="red", alpha=0.8, width=0.8, zorder=4)

    patches = [
        mpatches.Patch(color="grey",  alpha=0.6, label="SCE accuracy"),
        mpatches.Patch(color="green", alpha=0.8, label="HCE better"),
        mpatches.Patch(color="red",   alpha=0.8, label="HCE worse"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True)

    ax.set_xlabel("Class (sorted by SCE accuracy, descending)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=90, fontsize=label_fontsize, ha="center")
    ax.tick_params(axis="x", pad=2)

    plt.tight_layout()
    matplotlib.rcParams['svg.fonttype'] = 'none'
    ax.set_clip_on(False)
    svg_out = save_path.replace('.pdf', '.svg')
    plt.savefig(svg_out, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    remove_clippath(svg_out)

    print(f"{title}")
    print(f"  HCE better: {imp.sum()} / {n}   HCE worse: {wor.sum()} / {n}")
    print(f"  Mean delta: {delta.mean():+.2f}%")
    print(f"  Saved: {save_path}\n")


plot_level(sce_leaf, hce_leaf, leaf_names,
           "ResNet-50 — Per-variant accuracy: SCE vs HCE",
           os.path.join(BASE_DIR, "per_leaf_resnet50_fgvc.pdf"),
           label_fontsize=6)

plot_level(sce_fam, hce_fam, family_names,
           "ResNet-50 — Per-family accuracy: SCE vs HCE",
           os.path.join(BASE_DIR, "per_family_resnet50_fgvc.pdf"),
           label_fontsize=8)

plot_level(sce_man, hce_man, manuf_names,
           "ResNet-50 — Per-manufacturer accuracy: SCE vs HCE",
           os.path.join(BASE_DIR, "per_manufacturer_resnet50_fgvc.pdf"),
           label_fontsize=10)