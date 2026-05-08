"""
resnet50_fgvc_sunburst.py

Radial sunburst showing per-node accuracy delta (HCE 0.2 adj to SCE) for
ResNet50 on FGVC Aircraft.

HCE model outputs 201 logits covering all non-root nodes:
    indices   0-101  : variants (leaves)
    indices 102-171  : families
    indices 172-200  : manufacturers (29 of 41 that appear in training)

  Leaf-level accuracy:
    softmax over all 201 outputs; argmax over all 201.
    If argmax >= 102 (not a leaf) that means it is automatically incorrect.

  Family-level probability for node F:
    P(F) = softmax[F_idx]                       (direct output for this family)
         + Σ softmax[v] for variant children v  (aggregate leaves up)

  Manufacturer-level probability for node M:
    P(M) = (softmax[M_idx])
         + Σ softmax[F_idx] for family children F
         + Σ softmax[v]     for variant descendants

SCE model outputs 102 logits (variants only):

  Leaf-level accuracy: standard argmax over 102.

  Family-level probability for node F:
    P(F) = Σ softmax[v] for variant children v   (aggregate leaves up)

  Manufacturer-level probability for node M:
    P(M) = Σ P(F) for family children F

Rings (inner to outer):  Manufacturer -> Family -> Variant
Node colour:  green = HCE better, red = SCE better (intensity proportional to magnitude)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "/data/user"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/fgvc")
DATA_DIR = os.path.join(BASE_DIR, "fgvc-aircraft-2013b/fgvc-aircraft-2013b/data")
OUT_PATH = os.path.join(CKPT_DIR, "resnet50_fgvc_sunburst.png")

device = (
    "cuda:7"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

NUM_SCE_CLASSES = 102
NUM_HCE_CLASSES = 201

def load_label_list(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

variants = load_label_list(f"{DATA_DIR}/variants.txt")
families = load_label_list(f"{DATA_DIR}/families.txt")
manufacturers = load_label_list(f"{DATA_DIR}/manufacturers.txt") # there are 41 manufacturers in this text file, but only 30 have images associated with them

VARIANT_OFFSET = 0
FAMILY_OFFSET = len(variants)                           
MANUFACTURER_OFFSET = FAMILY_OFFSET + len(families)          
ROOT_IDX = MANUFACTURER_OFFSET + len(manufacturers)

variant_to_idx = {v: i for i, v in enumerate(variants)}
family_to_idx = {f: i + FAMILY_OFFSET for i, f in enumerate(families)}
manufacturer_to_idx = {m: i + MANUFACTURER_OFFSET for i, m in enumerate(manufacturers)}

idx_to_name = {}
for v, i in variant_to_idx.items(): idx_to_name[i] = v
for f, i in family_to_idx.items(): idx_to_name[i] = f
for m, i in manufacturer_to_idx.items(): idx_to_name[i] = m
idx_to_name[ROOT_IDX] = "ROOT"

def load_image_labels(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

variant_labels = load_image_labels(f"{DATA_DIR}/images_variant_trainval.txt")
family_labels = load_image_labels(f"{DATA_DIR}/images_family_trainval.txt")
manufacturer_labels = load_image_labels(f"{DATA_DIR}/images_manufacturer_trainval.txt")

variant_to_family, family_to_manufacturer = {}, {}
for img_id, var in variant_labels.items():
    fam = family_labels.get(img_id)
    if var and fam:
        variant_to_family[var] = fam
for img_id, fam in family_labels.items():
    mfr = manufacturer_labels.get(img_id)
    if fam and mfr:
        family_to_manufacturer[fam] = mfr

child_to_parent = {}
for v, f in variant_to_family.items():
    if v in variant_to_idx and f in family_to_idx:
        child_to_parent[variant_to_idx[v]] = family_to_idx[f]
for f, m in family_to_manufacturer.items():
    if f in family_to_idx and m in manufacturer_to_idx:
        child_to_parent[family_to_idx[f]] = manufacturer_to_idx[m]
for m_idx in manufacturer_to_idx.values():
    child_to_parent[m_idx] = ROOT_IDX

parent_to_children = defaultdict(list)
for child, parent in child_to_parent.items():
    parent_to_children[parent].append(child)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class FGVCAircraftDataset(Dataset):
    def __init__(self, fgvc_dir, split="test", transform=None):
        self.transform = transform
        self.image_dir = os.path.join(fgvc_dir, "images")
        vlist = []
        with open(os.path.join(fgvc_dir, "variants.txt")) as f:
            for line in f:
                v = line.strip()
                if v:
                    vlist.append(v)
        v2i = {v: i for i, v in enumerate(vlist)}
        self.samples = []
        with open(os.path.join(fgvc_dir, f"images_variant_{split}.txt")) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    img_id, vname = parts
                    self.samples.append((
                        os.path.join(self.image_dir, f"{img_id}.jpg"),
                        v2i[vname]
                    ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

test_loader = DataLoader(
    FGVCAircraftDataset(DATA_DIR, split="test", transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])),
    batch_size=128, shuffle=False, num_workers=8, pin_memory=True,
)

def load_resnet50(num_classes, ckpt_path):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m = m.to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(state["model"])
    m.eval()
    return m

model_hce = load_resnet50(NUM_HCE_CLASSES,
    os.path.join(CKPT_DIR, "resnet50_hce_0.2_no_smooth_adj", "checkpoint.pth"))
model_sce = load_resnet50(NUM_SCE_CLASSES,
    os.path.join(CKPT_DIR, "resnet50_sce_no_smooth", "checkpoint.pth"))

all_probs_hce, all_probs_sce, all_true_labels = [], [], []

with torch.no_grad():
    for X, labels in test_loader:
        X = X.to(device)
        # HCE: full softmax over all 201 outputs
        all_probs_hce.append(torch.softmax(model_hce(X), dim=-1).cpu().numpy())
        # SCE: softmax over 102 leaf outputs
        all_probs_sce.append(torch.softmax(model_sce(X), dim=-1).cpu().numpy())
        all_true_labels.append(labels.numpy())

probs_hce = np.concatenate(all_probs_hce)    # (N_test, 201)
probs_sce = np.concatenate(all_probs_sce)    # (N_test, 102)
true_labels = np.concatenate(all_true_labels)  # (N_test,)

true_family = np.array([child_to_parent.get(int(l), -1) for l in true_labels])
true_mfr    = np.array([child_to_parent.get(int(f), -1) for f in true_family])

#Aggregated node probabilities
_hce_cache = {}
def agg_hce(node):
    if node not in _hce_cache:
        agg = np.zeros(len(true_labels), dtype=np.float32)
        agg += probs_hce[:, node]
        for child in parent_to_children[node]:
            agg += agg_hce(child)
        _hce_cache[node] = agg
    return _hce_cache[node]

_sce_cache = {}
def agg_sce(node):
    if node not in _sce_cache:
        if node < FAMILY_OFFSET: #need it to be a leaf node
            agg = probs_sce[:, node]
        else:
            agg = np.zeros(len(true_labels), dtype=np.float32)
            for child in parent_to_children[node]:
                agg += agg_sce(child)
        _sce_cache[node] = agg
    return _sce_cache[node]

all_nodes = (list(range(FAMILY_OFFSET))
             + list(family_to_idx.values())
             + list(manufacturer_to_idx.values()))

print("Pre-computing aggregated probabilities...")
for n in all_nodes:
    agg_hce(n)
    agg_sce(n)
print("Done.")

# Hierarchical Level-wise argmax predictions
def level_argmax(agg_fn, node_list):
    """Argmax over aggregated probabilities restricted to nodes at this level."""
    mat = np.stack([agg_fn(n) for n in node_list], axis=1)  # (N_test, |level|)
    return np.array(node_list)[mat.argmax(axis=1)]

# Leaf level — HCE: argmax over ALL 201 raw outputs;
raw_argmax_hce = probs_hce.argmax(axis=1)
pred_leaf_hce  = np.where(raw_argmax_hce < FAMILY_OFFSET,
                          raw_argmax_hce, -1)

# Leaf level — SCE: argmax over 102
pred_leaf_sce  = probs_sce.argmax(axis=1)

# Family level
f_list = sorted(family_to_idx.values())
pred_fam_hce = level_argmax(agg_hce, f_list)
pred_fam_sce = level_argmax(agg_sce, f_list)

# Manufacturer level
m_list = sorted(manufacturer_to_idx.values())
pred_mfr_hce = level_argmax(agg_hce, m_list)
pred_mfr_sce = level_argmax(agg_sce, m_list)

#Per-node accuracy
def node_acc(true_y, pred_y, node_idx):
    mask = true_y == node_idx
    if mask.sum() == 0:
        return np.nan
    return float((pred_y[mask] == node_idx).mean())

acc_hce, acc_sce = {}, {}
for v in range(FAMILY_OFFSET):
    acc_hce[v] = node_acc(true_labels, pred_leaf_hce, v)
    acc_sce[v] = node_acc(true_labels, pred_leaf_sce, v)
for f in family_to_idx.values():
    acc_hce[f] = node_acc(true_family, pred_fam_hce, f)
    acc_sce[f] = node_acc(true_family, pred_fam_sce, f)
for m in manufacturer_to_idx.values():
    acc_hce[m] = node_acc(true_mfr, pred_mfr_hce, m)
    acc_sce[m] = node_acc(true_mfr, pred_mfr_sce, m)

def delta(node):
    h, s = acc_hce.get(node, np.nan), acc_sce.get(node, np.nan)
    if np.isnan(h) or np.isnan(s):
        return 0.0
    return float(h - s)

#Colour map
all_deltas = [delta(n) for n in all_nodes]
max_abs    = max(abs(d) for d in all_deltas) + 1e-6

def node_color(d):
    t = np.clip(d / max_abs, -1.0, 1.0)
    if t >= 0:   # white → green
        return (1 - t * 0.55, 1.0, 1 - t * 0.55)
    else:        # white → red
        t = -t
        return (1.0, 1 - t * 0.55, 1 - t * 0.55)

#Layout helpers
def leaf_count(node):
    if node < FAMILY_OFFSET:
        return 1
    return sum(leaf_count(c) for c in parent_to_children[node])

def first_leaf(node):
    if node < FAMILY_OFFSET:
        return node
    return first_leaf(min(parent_to_children[node]))

sorted_manufacturers = sorted(manufacturer_to_idx.values(), key=first_leaf)
N_LEAVES = FAMILY_OFFSET
TWO_PI   = 2 * np.pi

RINGS = {
    "mfr":    (0.20, 0.22),
    "family": (0.44, 0.22),
    "leaf":   (0.68, 0.28),
}

fig, ax = plt.subplots(figsize=(22, 22), subplot_kw=dict(polar=True))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_axis_off()

def draw_wedge(ax, t1, t2, r_in, r_out, color, ew="white", lw=0.4, n=60):
    th = np.linspace(t1, t2, n)
    ax.fill(np.concatenate([th, th[::-1]]),
            np.concatenate([np.full(n, r_in), np.full(n, r_out)]),
            color=color, linewidth=lw, edgecolor=ew)

# Manufacturer ring
mfr_theta = {}
cur = 0.0
for m_idx in sorted_manufacturers:
    span = (leaf_count(m_idx) / N_LEAVES) * TWO_PI
    t1, t2 = cur, cur + span
    mfr_theta[m_idx] = (t1, t2)
    r_in, rw = RINGS["mfr"]
    draw_wedge(ax, t1, t2, r_in, r_in + rw, node_color(delta(m_idx)), lw=0.8)
    if span > 0.06:
        mid_t = (t1 + t2) / 2
        ax.text(mid_t, r_in + rw / 2, idx_to_name.get(m_idx, ""),
                ha="center", va="center", fontsize=5.5,
                rotation=np.degrees(mid_t) - 90, rotation_mode="anchor",
                fontweight="bold", color="#222")
    cur += span

# Family ring
fam_theta = {}
for m_idx in sorted_manufacturers:
    fam_cur = mfr_theta[m_idx][0]
    for f_idx in sorted(parent_to_children[m_idx], key=first_leaf):
        span = (leaf_count(f_idx) / N_LEAVES) * TWO_PI
        t1, t2 = fam_cur, fam_cur + span
        fam_theta[f_idx] = (t1, t2)
        r_in, rw = RINGS["family"]
        draw_wedge(ax, t1, t2, r_in, r_in + rw, node_color(delta(f_idx)), lw=0.5)
        fam_cur += span

# Leaf ring
for m_idx in sorted_manufacturers:
    for f_idx in sorted(parent_to_children[m_idx], key=first_leaf):
        v_cur = fam_theta[f_idx][0]
        for v_idx in sorted(parent_to_children[f_idx]):
            span = (1 / N_LEAVES) * TWO_PI
            t1, t2 = v_cur, v_cur + span
            r_in, rw = RINGS["leaf"]
            draw_wedge(ax, t1, t2, r_in, r_in + rw, node_color(delta(v_idx)), lw=0.2)
            v_cur += span

# Manufacturer boundary lines
r_in_mfr   = RINGS["mfr"][0]
r_out_leaf = RINGS["leaf"][0] + RINGS["leaf"][1]
for t1, _ in mfr_theta.values():
    ax.plot([t1, t1], [r_in_mfr, r_out_leaf], color="#555", lw=0.9, alpha=0.7)

# Ring labels
for ring_name, (r_in, rw) in RINGS.items():
    lbl = {"mfr": "Manufacturer", "family": "Family", "leaf": "Variant"}[ring_name]
    ax.text(-0.05, r_in + rw / 2, lbl, ha="right", va="center",
            fontsize=9, color="#444", fontweight="bold")

# Legend
def lpatch(frac, label):
    return mpatches.Patch(facecolor=node_color(frac * max_abs),
                          edgecolor="#aaa", linewidth=0.5, label=label)

ax.legend(
    handles=[
        lpatch( 1.0, f"HCE better  (+{max_abs*100:.1f}% max)"),
        lpatch( 0.5, f"HCE better  (+{max_abs*50:.1f}%)"),
        lpatch( 0.0, "No change"),
        lpatch(-0.5, f"SCE better  ({-max_abs*50:.1f}%)"),
        lpatch(-1.0, f"SCE better  ({-max_abs*100:.1f}% max)"),
    ],
    loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=10,
    title="Delta Accuracy (HCE - SCE)", title_fontsize=11,
    framealpha=0.9, edgecolor="#ccc",
)

ax.set_title(
    "ResNet50 FGVC Aircraft — Per-Node Accuracy Delta\n"
    "HCE 0.2 adj  vs  SCE  |  Rings: Manufacturer -> Family -> Variant",
    fontsize=15, pad=28, fontweight="bold", color="#111",
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUT_PATH}")
for level, nodes in [
    ("Manufacturer", m_list),
    ("Family",       f_list),
    ("Variant",      list(range(FAMILY_OFFSET))),
]:
    finite = [delta(n) for n in nodes if not np.isnan(delta(n))]
    print(f"{level:15s}  mean delta = {np.mean(finite):+.3f}  "
          f"HCE better={sum(d > 0 for d in finite)}/{len(nodes)}  "
          f"SCE better={sum(d < 0 for d in finite)}/{len(nodes)}")

non_leaf_rate = (raw_argmax_hce >= FAMILY_OFFSET).mean()
print(f"\nHCE non-leaf argmax rate: {100*non_leaf_rate:.1f}% "
      f"(these are auto-incorrect at leaf level)")