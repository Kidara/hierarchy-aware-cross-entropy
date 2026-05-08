import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
# from resnet18_train_hce import test_loader
import os
from pathlib import Path

BASE_DIR = "/data/user"
os.makedirs(BASE_DIR, exist_ok=True)

device = (
    "cuda:7"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# checkpoint_path_sce = os.path.join(BASE_DIR, "checkpoints", "resnet50_sce_no_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_sce_smooth = os.path.join(BASE_DIR, "checkpoints", "resnet50_sce_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_sce_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_sce_no_smooth_adj", "checkpoint_best_accuracy.pth")
# checkpoint_path_sce_smooth_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_sce_smooth_adj", "checkpoint_best_accuracy.pth")

# checkpoint_path_hce_2 = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.2_no_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_2_smooth = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.2_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_2_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.2_no_smooth_adj", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_2_smooth_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.2_smooth_adj", "checkpoint_best_accuracy.pth")

# checkpoint_path_hce_5 = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.5_no_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_5_smooth = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.5_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_5_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.5_no_smooth_adj", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_5_smooth_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.5_smooth_adj", "checkpoint_best_accuracy.pth")

# checkpoint_path_hce_75 = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.75_no_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_75_smooth = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.75_smooth", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_75_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.75_no_smooth_adj", "checkpoint_best_accuracy.pth")
# checkpoint_path_hce_75_smooth_adj = os.path.join(BASE_DIR, "checkpoints", "resnet50_hce_0.75_smooth_adj", "checkpoint_best_accuracy.pth")

def create_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def compute_overall_top1_top5(dataloader, model_hce, model_sce, k=5):
    model_hce.eval()
    model_sce.eval()
    
    total = 0
    correct1_hce = 0
    correct5_hce = 0
    correct1_sce = 0
    correct5_sce = 0
    
    with torch.no_grad():
        for X, y_fine, y_coarse in dataloader:
            X = X.to(device)
            y_fine = y_fine.to(device)
            y_true = y_fine - 20
            
            logits_hce = model_hce(X)
            pred1_hce = logits_hce.argmax(1)
            pred1_hce = torch.where(pred1_hce >= 20, pred1_hce - 20, -1)
            topk_hce = torch.topk(logits_hce, k=k, dim=1).indices
            topk_hce = torch.where(topk_hce >= 20, topk_hce - 20, -1)
            
            logits_sce = model_sce(X)
            pred1_sce = logits_sce.argmax(1)
            topk_sce = torch.topk(logits_sce, k=k, dim=1).indices

            correct1_hce += (pred1_hce == y_true).sum().item()
            correct1_sce += (pred1_sce == y_true).sum().item()
            
            for i in range(len(y_true)):
                if y_true[i] in topk_hce[i]:
                    correct5_hce += 1
                if y_true[i] in topk_sce[i]:
                    correct5_sce += 1
            
            total += len(y_true)
    
    acc1_hce = 100 * correct1_hce / total
    acc5_hce = 100 * correct5_hce / total
    acc1_sce = 100 * correct1_sce / total
    acc5_sce = 100 * correct5_sce / total
    
    return acc1_hce, acc5_hce, acc1_sce, acc5_sce

# model_sce = create_resnet50(100).to(device)
# model_sce_smooth = create_resnet50(100).to(device)
# model_sce_adj = create_resnet50(100).to(device)
# model_sce_smooth_adj = create_resnet50(100).to(device)

# ckpt_sce = torch.load(checkpoint_path_sce, map_location=device, weights_only=False)
# model_sce.load_state_dict(ckpt_sce["model"])

# ckpt_sce_smooth = torch.load(checkpoint_path_sce_smooth, map_location=device, weights_only=False)
# model_sce_smooth.load_state_dict(ckpt_sce_smooth["model"])

# ckpt_sce_adj = torch.load(checkpoint_path_sce_adj, map_location=device, weights_only=False)
# model_sce_adj.load_state_dict(ckpt_sce_adj["model"])

# ckpt_sce_smooth_adj = torch.load(checkpoint_path_sce_smooth_adj, map_location=device, weights_only=False)
# model_sce_smooth_adj.load_state_dict(ckpt_sce_smooth_adj["model"])

# checkpoints_hce = [
#     ("HCE 0.2", checkpoint_path_hce_2, model_sce_adj),
#     ("HCE 0.2 adj", checkpoint_path_hce_2_adj, model_sce),
#     ("HCE 0.2 smooth", checkpoint_path_hce_2_smooth, model_sce_smooth_adj),
#     ("HCE 0.2 smooth adj", checkpoint_path_hce_2_smooth_adj, model_sce_smooth),
#     ("HCE 0.5", checkpoint_path_hce_5, model_sce_adj),
#     ("HCE 0.5 adj", checkpoint_path_hce_5_adj, model_sce),
#     ("HCE 0.5 smooth", checkpoint_path_hce_5_smooth, model_sce_smooth_adj),
#     ("HCE 0.5 smooth adj", checkpoint_path_hce_5_smooth_adj, model_sce_smooth),
#     ("HCE 0.75", checkpoint_path_hce_75, model_sce_adj),
#     ("HCE 0.75 adj", checkpoint_path_hce_75_adj, model_sce),
#     ("HCE 0.75 smooth", checkpoint_path_hce_75_smooth, model_sce_smooth_adj),
#     ("HCE 0.75 smooth adj", checkpoint_path_hce_75_smooth_adj, model_sce_smooth),
# ]

results = []
# accuracy for resnet18
# accuracy = {"HCE 0.2": [64.35, 84.19, 51.17, 73.75],
#             "HCE 0.2 adj": [67.76, 87.09, 59.03, 79.85],
#             "HCE 0.5": [60.96, 81.66, 58.93, 78.52],
#             "HCE 0.5 adj": [63.21, 83.5, 59.03, 79.85],
#             "HCE all one": [61.56, 82.28, 59.03, 79.85]}

# accuracy for resnet34
accuracy = {"HCE 0.2": [69.2364, 87.6913, 57.8898, 79.6046],
            "HCE 0.2 adj": [71.6681, 89.4828, 61.6283, 82.2527],
            "HCE 0.5": [65.3554, 84.8038, 61.1792, 81.8438],
            "HCE 0.5 adj": [64.17, 84.1871, 61.6283, 82.2527],
            "HCE all one": [57.6597, 79.5474, 61.6283, 82.2527]}

for name in accuracy:
    # model_hce = create_resnet50(120).to(device)
    # ckpt_hce = torch.load(ckpt_path, map_location=device, weights_only=False)
    # model_hce.load_state_dict(ckpt_hce["model"])
    
    # acc1_hce, acc5_hce, acc1_sce, acc5_sce = compute_overall_top1_top5(test_loader, model_hce, sce_model, k=5)
    acc1_hce, acc5_hce, acc1_sce, acc5_sce = accuracy[name]
    results.append((name, acc1_hce, acc5_hce, acc1_sce, acc5_sce))
    print(f"{name}: Top-1 HCE {acc1_hce:.2f}% vs SCE {acc1_sce:.2f}% (Δ {acc1_hce - acc1_sce:+.2f}%), Top-5 HCE {acc5_hce:.2f}% vs SCE {acc5_sce:.2f}% (Δ {acc5_hce - acc5_sce:+.2f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))


x = np.arange(len(results))
width = 0.6

for ax, metric_idx, title in [(ax1, 0, 'Resnet18 Top-1 Accuracy'), (ax2, 1, 'Resnet18 Top-5 Accuracy')]:
    for i, (name, acc1_hce, acc5_hce, acc1_sce, acc5_sce) in enumerate(results):
        hce_val = acc1_hce if metric_idx == 0 else acc5_hce
        sce_val = acc1_sce if metric_idx == 0 else acc5_sce
        
        ax.bar(i, sce_val, width, color='lightgrey')
        
        diff = hce_val - sce_val
        if diff > 0:
            ax.bar(i, diff, width, bottom=sce_val, color='green')
        else:
            ax.bar(i, -diff, width, bottom=hce_val, color='red')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in results], rotation=45, ha='right')
    ax.set_ylim(45, 90)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "g"), dpi=300)
plt.close()