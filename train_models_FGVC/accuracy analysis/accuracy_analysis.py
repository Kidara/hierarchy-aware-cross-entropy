import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
from swinT_hce import test_loader
import os

#NEED TO REDO VIT_SCE_0.2_NO_SMOOTH_ADJ

BASE_DIR = "/data/user/checkpoints/fgvc"

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

NUM_SCE_CLASSES = 102
NUM_HCE_CLASSES = 201

def ckpt(name, filename="checkpoint_best_top1.pth"):
    return os.path.join(BASE_DIR, name, filename)

checkpoint_path_sce         = ckpt("resnet18_sce_no_smooth")
checkpoint_path_sce_0_2_adj = ckpt("resnet18_sce_no_smooth")
checkpoint_path_sce_0_5_adj = ckpt("resnet18_sce_0.5_no_smooth_adj")

checkpoint_path_hce_2       = ckpt("resnet18_hce_0.2_no_smooth")
checkpoint_path_hce_2_adj   = ckpt("resnet18_hce_0.2_no_smooth_adj")
checkpoint_path_hce_5       = ckpt("resnet18_hce_0.5_no_smooth")
checkpoint_path_hce_5_adj   = ckpt("resnet18_hce_0.5_no_smooth_adj")
checkpoint_path_hce_all_one = ckpt("resnet18_hce_all_one_no_smooth")

def create_resnet18(num_classes):
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(num_classes, path):
    model = create_resnet18(num_classes).to(device)
    ckpt_data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model"])
    return model

def compute_overall_top1_top5(dataloader, model_hce, model_sce, k=5):
    model_hce.eval()
    model_sce.eval()

    total = 0
    correct1_hce = correct5_hce = 0
    correct1_sce = correct5_sce = 0

    with torch.no_grad():
        for X, labels in dataloader:
            X      = X.to(device)
            labels = labels.to(device)

            logits_hce = model_hce(X)
            pred1_hce  = logits_hce.argmax(1)
            topk_hce   = torch.topk(logits_hce, k=k, dim=1).indices

            logits_sce = model_sce(X)
            pred1_sce  = logits_sce.argmax(1)
            topk_sce   = torch.topk(logits_sce, k=k, dim=1).indices

            correct1_hce += (pred1_hce == labels).sum().item()
            correct1_sce += (pred1_sce == labels).sum().item()

            for i in range(len(labels)):
                if labels[i] in topk_hce[i]: correct5_hce += 1
                if labels[i] in topk_sce[i]: correct5_sce += 1

            total += len(labels)

    return (100 * correct1_hce / total, 100 * correct5_hce / total,
            100 * correct1_sce / total, 100 * correct5_sce / total)

model_sce         = load_model(NUM_SCE_CLASSES, checkpoint_path_sce)
model_sce_0_2_adj = load_model(NUM_SCE_CLASSES, checkpoint_path_sce_0_2_adj)
model_sce_0_5_adj = load_model(NUM_SCE_CLASSES, checkpoint_path_sce_0_5_adj)

checkpoints_hce = [
    ("HCE 0.2",     checkpoint_path_hce_2,       model_sce_0_2_adj),
    ("HCE 0.2 adj", checkpoint_path_hce_2_adj,   model_sce),
    ("HCE 0.5",     checkpoint_path_hce_5,       model_sce_0_5_adj),
    ("HCE 0.5 adj", checkpoint_path_hce_5_adj,   model_sce),
    ("HCE all_one", checkpoint_path_hce_all_one, model_sce),
]

results = []
for name, ckpt_path, sce_model in checkpoints_hce:
    model_hce = load_model(NUM_HCE_CLASSES, ckpt_path)
    acc1_hce, acc5_hce, acc1_sce, acc5_sce = compute_overall_top1_top5(
        test_loader, model_hce, sce_model, k=5
    )
    results.append((name, acc1_hce, acc5_hce, acc1_sce, acc5_sce))
    print(f"{name}: Top-1 HCE {acc1_hce:.2f}% vs SCE {acc1_sce:.2f}% "
          f"(Δ {acc1_hce - acc1_sce:+.2f}%), "
          f"Top-5 HCE {acc5_hce:.2f}% vs SCE {acc5_sce:.2f}% "
          f"(Δ {acc5_hce - acc5_sce:+.2f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(results))
width = 0.6

for ax, metric_idx, title in [
    (ax1, 0, "resnet18 FGVC Top-1 Accuracy"),
    (ax2, 1, "resnet18 FGVC Top-5 Accuracy"),
]:
    for i, (name, acc1_hce, acc5_hce, acc1_sce, acc5_sce) in enumerate(results):
        hce_val = acc1_hce if metric_idx == 0 else acc5_hce
        sce_val = acc1_sce if metric_idx == 0 else acc5_sce

        ax.bar(i, sce_val, width, color="lightgrey")
        diff = hce_val - sce_val
        ax.bar(i, abs(diff), width, bottom=min(hce_val, sce_val),
               color="green" if diff > 0 else "red")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in results], rotation=45, ha="right")
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resnet18_fgvc_comparison.png"), dpi=300)
plt.close()