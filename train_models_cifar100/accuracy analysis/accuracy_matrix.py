import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CIFAR100_class import CIFAR100Dataset

BASE_DIR  = "/data/user/checkpoints/cifar100"
CKPT_FILE = "checkpoint.pth"
def _unpickle(file):
    with open(file, "rb") as f:
        return pickle.load(f, encoding="bytes")
_test_data = _unpickle("cifar-100-python/test")
CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_STD  = [0.2675, 0.2565, 0.2761]

NUM_SCE = 100
NUM_HCE = 120
MISSING = "MISSING"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_resnet18(n):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n)
    return model

def make_resnet34(n):
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n)
    return model

def make_resnet50(n):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n)
    return model

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

test_loader = DataLoader(
    CIFAR100Dataset(_test_data, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]), hce=True),
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
)
 
# 224x224 loader -- ViT was trained on images resized to 224
vit_test_loader = DataLoader(
    CIFAR100Dataset(_test_data, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]), hce=True),
    batch_size=64, shuffle=False, num_workers=4, pin_memory=True,
)  


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
        "hce_0.2":     "vit_hce_0.2_no_smooth2",
        "hce_0.2_adj": "vit_hce_0.2_no_smooth_adj2",
        "hce_0.5":     "vit_hce_0.5_no_smooth2",
        "hce_0.5_adj": "vit_hce_0.5_no_smooth_adj2",
        "hce_0.7":     "vit_hce_0.7_no_smooth2",
        "hce_0.7_adj": "vit_hce_0.7_no_smooth_adj2",
        "sce":         "vit_sce_no_smooth2",
        "sce_0.2_adj": "vit_sce_0.2_no_smooth_adj2",
        "sce_0.5_adj": "vit_sce_0.5_no_smooth_adj2",
        "sce_0.7_adj": "vit_sce_0.7_no_smooth_adj2",
    }, vit_test_loader),
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
def evaluate(model_hce, model_sce, loader, k=5):
    """Returns (top1_hce, top5_hce, top1_sce, top5_sce). None if either model missing."""
    if model_hce is None or model_sce is None:
        return None

    total = correct1_hce = correct5_hce = correct1_sce = correct5_sce = 0

    for X, y_fine, y_coarse in loader:
        X      = X.to(device)
        y_fine = y_fine.to(device)
        y_true = y_fine - 20 

        logits_hce = model_hce(X)
        pred1_hce  = logits_hce.argmax(1)
        pred1_hce  = torch.where(pred1_hce >= 20, pred1_hce - 20, torch.tensor(-1, device=device))
        topk_hce   = torch.topk(logits_hce, k=k, dim=1).indices
        topk_hce   = torch.where(topk_hce >= 20, topk_hce - 20, torch.tensor(-1, device=device))

        logits_sce = model_sce(X)
        pred1_sce  = logits_sce.argmax(1)
        topk_sce   = torch.topk(logits_sce, k=k, dim=1).indices

        correct1_hce += (pred1_hce == y_true).sum().item()
        correct1_sce += (pred1_sce == y_true).sum().item()

        for i in range(len(y_true)):
            if y_true[i] in topk_hce[i]: correct5_hce += 1
            if y_true[i] in topk_sce[i]: correct5_sce += 1

        total += len(y_true)

    return (100 * correct1_hce / total, 100 * correct5_hce / total,
            100 * correct1_sce  / total, 100 * correct5_sce  / total)


arch_names = [cfg[0] for cfg in ARCH_CONFIGS]
col_labels = [c[0] for c in COMPARISONS]

delta1_matrix = np.full((len(ARCH_CONFIGS), len(COMPARISONS)), np.nan)
delta5_matrix = np.full((len(ARCH_CONFIGS), len(COMPARISONS)), np.nan)

for row_idx, arch_cfg in enumerate(ARCH_CONFIGS):
    arch_name, make_fn, folders = arch_cfg[0], arch_cfg[1], arch_cfg[2]
    loader = arch_cfg[3] if len(arch_cfg) > 3 else test_loader
    print(f"\n{'='*60}\n{arch_name}\n{'='*60}")
 
    sce_cache = {}
    for key in ("sce", "sce_0.2_adj", "sce_0.5_adj", "sce_0.7_adj"):
        sce_cache[key] = load_model(make_fn, NUM_SCE, folders[key])
 
    for col_idx, (label, hce_key, sce_key) in enumerate(COMPARISONS):
        model_hce = load_model(make_fn, NUM_HCE, folders[hce_key])
        model_sce = sce_cache[sce_key]
 
        result = evaluate(model_hce, model_sce, loader)
        if result is None:
            print(f"  {label:12s}: MISSING")
        else:
            t1h, t5h, t1s, t5s = result
            d1, d5 = t1h - t1s, t5h - t5s
            delta1_matrix[row_idx, col_idx] = d1
            delta5_matrix[row_idx, col_idx] = d5
            print(f"  {label:12s}: Top-1 HCE {t1h:.2f}% vs SCE {t1s:.2f}% (Δ{d1:+.2f}%)  "
                  f"Top-5 HCE {t5h:.2f}% vs SCE {t5s:.2f}% (Δ{d5:+.2f}%)")


def plot_matrix(matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
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

plot_matrix(delta1_matrix, "CIFAR-100 — Top-1 Δ (HCE − SCE)", os.path.join(BASE_DIR, "matrix_cifar100_top1.png"))
plot_matrix(delta5_matrix, "CIFAR-100 — Top-5 Δ (HCE − SCE)", os.path.join(BASE_DIR, "matrix_cifar100_top5.png"))