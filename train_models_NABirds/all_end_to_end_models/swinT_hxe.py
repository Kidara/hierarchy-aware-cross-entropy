import os
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",   default="5")
parser.add_argument("--alpha", type=float, default=0.5, choices=[0.2, 0.5])
args   = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

BASE_DIR  = "/data/user"
NAB_DIR   = os.path.join(BASE_DIR, "nabirds")
ALPHA_TAG = str(args.alpha).replace(".", "")
CKPT_DIR  = os.path.join(BASE_DIR, f"checkpoints/NA_Birds/swinT_hxe_{ALPHA_TAG}")
TB_DIR    = os.path.join(BASE_DIR, f"resnet_NABirds/swinT_hxe_{ALPHA_TAG}")

FEAT_DIM  = 512
BATCH     = 128
EPOCHS    = 500
LR        = 1e-3 * (128 / 1024)
MOMENTUM  = 0.9
WD        = 0.05
WORKERS   = 8
N_TOTAL      = 1011
leaf_df      = pd.read_csv("leaf_index.csv", index_col=0)
leaf_list    = leaf_df["original_node_id"].tolist()
N_LEAVES     = len(leaf_list)   # ~555

orig_to_leaf = torch.full((N_TOTAL,), -1, dtype=torch.long)
for leaf_idx, orig_id in enumerate(leaf_list):
    orig_to_leaf[orig_id] = leaf_idx

print(f"N_LEAVES={N_LEAVES}")
child_leaf_masks  = torch.load("child_leaf_masks.pt").float().to(device)
parent_leaf_masks = torch.load("parent_leaf_masks.pt").float().to(device)
N_LEAVES_check, E, _ = child_leaf_masks.shape
assert N_LEAVES_check == N_LEAVES, "Mask shape/leaf_index mismatch"

depth_matrix = torch.from_numpy(
    pd.read_csv("depth_matrix.csv", index_col=0).values.astype(np.float32)
).to(device)

edge_weights = torch.exp(-args.alpha * depth_matrix)

print(f"child_leaf_masks: {child_leaf_masks.shape}, E={E}")

S_full = torch.from_numpy(
    pd.read_csv("sibling_matrix.csv", index_col=0).values.astype(np.float32)
)                                                             # [1011, 1011]
S = S_full[leaf_list][:, leaf_list].to(device)

class HXELoss(nn.Module):
    def __init__(self, child_masks, parent_masks, weights):
        super().__init__()
        self.register_buffer("child_masks",  child_masks)
        self.register_buffer("parent_masks", parent_masks)
        self.register_buffer("weights",      weights)

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        c_masks = self.child_masks[targets]
        p_masks = self.parent_masks[targets]
        w       = self.weights[targets]

        child_mass  = (c_masks * probs.unsqueeze(1)).sum(dim=-1)
        parent_mass = (p_masks * probs.unsqueeze(1)).sum(dim=-1)

        cond      = child_mass / parent_mass.clamp_min(1e-12)
        edge_loss = -torch.log(cond.clamp_min(1e-12))
        return (w * edge_loss).sum(dim=1).mean()


criterion = HXELoss(child_leaf_masks, parent_leaf_masks, edge_weights)


def verify_mass_conservation(loss_module, n_leaves, dev):
    uniform     = torch.full((1, n_leaves), 1.0 / n_leaves, device=dev)
    all_targets = torch.arange(n_leaves, device=dev)
    c_masks     = loss_module.child_masks[all_targets]
    p_masks     = loss_module.parent_masks[all_targets]
    child_mass  = (c_masks * uniform).sum(dim=-1)
    parent_mass = (p_masks * uniform).sum(dim=-1)
    if not (child_mass <= parent_mass + 1e-6).all():
        bad = (child_mass > parent_mass + 1e-6).nonzero(as_tuple=False)
        raise RuntimeError(f"Mass conservation violated (child > parent) at {bad.tolist()}")
    if not (parent_mass <= 1.0 + 1e-6).all():
        bad = (parent_mass > 1.0 + 1e-6).nonzero(as_tuple=False)
        raise RuntimeError(f"Mass conservation violated (parent > 1) at {bad.tolist()}")
    print("Mass conservation check passed.")

verify_mass_conservation(criterion, N_LEAVES, device)


class NABirdsDataset(Dataset):
    def __init__(self, nabirds_dir, split="train", transform=None):
        self.transform = transform

        split_map = {}
        with open(os.path.join(nabirds_dir, "train_test_split.txt")) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                split_map[img_id] = int(is_train)

        image_paths = {}
        with open(os.path.join(nabirds_dir, "images.txt")) as f:
            for line in f:
                img_id, path = line.strip().split()
                image_paths[img_id] = path

        image_labels = {}
        with open(os.path.join(nabirds_dir, "image_class_labels.txt")) as f:
            for line in f:
                img_id, label = line.strip().split()
                image_labels[img_id] = int(label)

        is_train = (split == "train")
        self.samples = []
        for img_id, flag in split_map.items():
            if bool(flag) == is_train:
                orig_label = image_labels[img_id]
                leaf_idx   = orig_to_leaf[orig_label].item()
                assert leaf_idx >= 0, \
                    f"Label {orig_label} (img {img_id}) is not a leaf node"
                path = os.path.join(nabirds_dir, "images", image_paths[img_id])
                self.samples.append((path, leaf_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, leaf_idx = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, leaf_idx


MEAN = [125.30513277 / 255, 129.66606421 / 255, 118.45121113 / 255]
STD  = [57.0045467  / 255, 56.70059436  / 255, 68.44430446  / 255]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

train_dataset = NABirdsDataset(NAB_DIR, split="train", transform=train_transform)
test_dataset  = NABirdsDataset(NAB_DIR, split="test",  transform=test_transform)
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH,
                          shuffle=True,  num_workers=WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH,
                          shuffle=False, num_workers=WORKERS, pin_memory=True)


model = models.swin_t(weights=None)
model.head = nn.Linear(model.head.in_features, N_LEAVES)
model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD,
    betas=(0.9, 0.999),
)

warmup_epochs = 20
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(max(1, EPOCHS - warmup_epochs))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def metrics(logits, targets):
    pred1    = logits.argmax(1)
    c1       = (pred1 == targets).sum().item()
    _, top5  = logits.topk(5, dim=1)
    c5       = top5.eq(targets.unsqueeze(1)).sum().item()
    cs       = S[targets.cpu(), pred1.cpu()].sum().item()
    return c1, c5, cs

def train_epoch(loader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    t0 = time.time()
    rl, ns, nb, c1t, c5t, cst = 0.0, 0, 0, 0, 0, 0

    for batch_idx, (X, targets) in enumerate(loader):
        X, targets = X.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        B = len(X); ns += B; nb += 1; rl += loss.item()
        with torch.no_grad():
            c1, c5, cs = metrics(logits, targets)
        c1t += c1; c5t += c5; cst += cs

        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx}, loss: {loss.item():.4f}, "
                  f"elapsed: {time.time()-t0:.1f}s")

    avg_loss = rl / nb
    writer.add_scalar("Train/loss",        avg_loss,       epoch)
    writer.add_scalar("Train/top1",        100 * c1t / ns, epoch)
    writer.add_scalar("Train/top5",        100 * c5t / ns, epoch)
    writer.add_scalar("Train/sibling_acc", 100 * cst / ns, epoch)
    print(f"  [train] loss={avg_loss:.4f}  top1={100*c1t/ns:.1f}%  "
          f"top5={100*c5t/ns:.1f}%  sibling={100*cst/ns:.1f}%  "
          f"({time.time()-t0:.1f}s)")
    return {"train_loss": avg_loss, "train_top1": c1t/ns,
            "train_top5": c5t/ns,   "train_sibling_acc": cst/ns}


def eval_epoch(loader, model, loss_fn, epoch, writer):
    model.eval()
    rl, ns, nb, c1t, c5t, cst = 0.0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, targets in loader:
            X, targets = X.to(device), targets.to(device)
            logits = model(X)
            rl    += loss_fn(logits, targets).item()
            B = len(X); ns += B; nb += 1
            c1, c5, cs = metrics(logits, targets)
            c1t += c1; c5t += c5; cst += cs

    avg_loss = rl / nb
    writer.add_scalar("Val/loss",        avg_loss,       epoch)
    writer.add_scalar("Val/top1",        100 * c1t / ns, epoch)
    writer.add_scalar("Val/top5",        100 * c5t / ns, epoch)
    writer.add_scalar("Val/sibling_acc", 100 * cst / ns, epoch)
    print(f"  [val]   loss={avg_loss:.4f}  top1={100*c1t/ns:.1f}%  "
          f"top5={100*c5t/ns:.1f}%  sibling={100*cst/ns:.1f}%")
    return {"val_loss": avg_loss, "val_top1": c1t/ns,
            "val_top5": c5t/ns,   "val_sibling_acc": cst/ns}


Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(TB_DIR)

best_top1, best_top5, best_val_loss = -1.0, -1.0, float("inf")


# CKPT_DIR  = os.path.join(BASE_DIR, f"checkpoints/NA_Birds/resnet50_hxe_{ALPHA_TAG}")
# best_top1 = -1.0
# best_top5 = -1.0
# best_val_loss = float("inf")
# start_epoch = 0
# checkpoint_path = os.path.join(BASE_DIR, "checkpoints/NA_Birds/resnet50_hxe_{ALPHA_TAG}", "checkpoint.pth") #check if checkpoint_path exists so we restore where training left off
# if os.path.exists(checkpoint_path):
#     print(f"Loading checkpoint...")
#     ckpt = torch.load(checkpoint_path)
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     scheduler.load_state_dict(ckpt["lr_scheduler"])
#     start_epoch = ckpt["epoch"] + 1

#     if "metrics" in ckpt: #restore best metrics so we can save future best ones correctly
#         best_top1 = ckpt["metrics"]["val_top1"]
#         best_val_loss = ckpt["metrics"]["val_loss"]
#         best_top5 = ckpt["metrics"]["val_top5"]

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}  [alpha={args.alpha}]")

    train_m = train_epoch(train_loader, model, criterion, optimizer, epoch, writer)
    scheduler.step()
    val_m   = eval_epoch(test_loader,  model, criterion,           epoch, writer)

    ckpt = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "alpha":     args.alpha,
        "n_leaves":  N_LEAVES,
        "leaf_list": leaf_list,
        "metrics":   {**train_m, **val_m},
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