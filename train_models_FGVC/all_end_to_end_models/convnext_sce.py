import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

BASE_DIR   = "/data/user"
FGVC_DIR  = os.path.join(BASE_DIR, "fgvc-aircraft-2013b/fgvc-aircraft-2013b/data")
os.makedirs(BASE_DIR, exist_ok=True)

class Params:
    def __init__(self):
        self.batch_size    = 128
        self.name          = "convnext_sce_0.7_no_smooth_adj"
        self.workers       = 8
        self.lr            = 4e-3 * (128 / 4096) / (10 / 7)
        self.momentum      = 0.9
        self.weight_decay  = 0.05
        self.num_classes   = 102
        self.label_smoothing = 0
        self.epochs        = 400

    def __repr__(self):
        return str(self.__dict__)

device = (
    "cuda:4"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

params = Params()
sibling_matrix = pd.read_csv('fgvc_sibling_matrix.csv', index_col=0)
S = torch.from_numpy(sibling_matrix.values[:params.num_classes, :params.num_classes].astype(np.float32)).to(device)

class FGVCAircraftDataset(Dataset):
    def __init__(self, fgvc_dir, split="train", transform=None):
        self.transform = transform
        self.image_dir = os.path.join(fgvc_dir, "images")

        variants = []
        with open(os.path.join(fgvc_dir, "variants.txt")) as f:
            for line in f:
                v = line.strip()
                if v:
                    variants.append(v)
        self.variant_to_idx = {v: i for i, v in enumerate(variants)}

        split_file_map = {
            "train":    "images_variant_train.txt",
            "val":      "images_variant_val.txt",
            "trainval": "images_variant_trainval.txt",
            "test":     "images_variant_test.txt",
        }

        annotation_file = os.path.join(fgvc_dir, split_file_map[split])
        self.samples = []
        with open(annotation_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    img_id, variant_name = parts
                    label    = self.variant_to_idx[variant_name]
                    img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

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

train_dataset = FGVCAircraftDataset(FGVC_DIR, split="trainval", transform=train_transform)
test_dataset  = FGVCAircraftDataset(FGVC_DIR, split="test",     transform=test_transform)
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                          shuffle=True,  num_workers=params.workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=params.batch_size,
                          shuffle=False, num_workers=params.workers, pin_memory=True)

model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, params.num_classes)
model = model.to(device)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)

def sibling_accuracy(pred_top1, targets):
    correct    = (pred_top1 == targets)
    is_sibling = S[targets, pred_top1].bool()
    return (correct | is_sibling).sum().item()

def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    running_loss    = 0.0
    correct_top1    = 0
    correct_top5    = 0
    correct_sibling = 0
    samples_seen    = 0
    num_batches     = 0
    start           = time.time()

    for batch, (X, labels) in enumerate(dataloader):
        X, labels = X.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        samples_seen += len(X)
        running_loss += loss.item()
        num_batches  += 1

        with torch.no_grad():
            pred_top1 = pred.argmax(1)
            correct_top1    += (pred_top1 == labels).sum().item()
            _, pred_top5     = pred.topk(5, dim=1)
            correct_top5    += pred_top5.eq(labels.unsqueeze(1)).sum().item()
            correct_sibling += sibling_accuracy(pred_top1, labels)

        if batch % 100 == 0:
            print(f"  batch {batch}, loss: {loss.item():.4f}, elapsed: {time.time()-start:.1f}s")

    avg_loss = running_loss / num_batches
    top1     = correct_top1    / samples_seen
    top5     = correct_top5    / samples_seen
    sib      = correct_sibling / samples_seen

    writer.add_scalar("Train/loss",        avg_loss,  epoch)
    writer.add_scalar("Train/top1",    100 * top1,    epoch)
    writer.add_scalar("Train/top5",    100 * top5,    epoch)
    writer.add_scalar("Train/sibling", 100 * sib,     epoch)
    print(f"Train — loss: {avg_loss:.4f}, top1: {100*top1:.1f}%, top5: {100*top5:.1f}%, sibling: {100*sib:.1f}%")
    return {"train_loss": avg_loss, "train_top1": top1, "train_top5": top5, "train_sibling": sib}


def test(dataloader, model, loss_fn, epoch, writer):
    model.eval()
    running_loss    = 0.0
    correct_top1    = 0
    correct_top5    = 0
    correct_sibling = 0
    samples_seen    = 0
    num_batches     = 0

    with torch.no_grad():
        for X, labels in dataloader:
            X, labels = X.to(device), labels.to(device)

            pred          = model(X)
            running_loss += loss_fn(pred, labels).item()
            num_batches  += 1
            samples_seen += len(X)

            pred_top1 = pred.argmax(1)
            correct_top1    += (pred_top1 == labels).sum().item()
            _, pred_top5     = pred.topk(5, dim=1)
            correct_top5    += pred_top5.eq(labels.unsqueeze(1)).sum().item()
            correct_sibling += sibling_accuracy(pred_top1, labels)

    avg_loss = running_loss / num_batches
    top1     = correct_top1    / samples_seen
    top5     = correct_top5    / samples_seen
    sib      = correct_sibling / samples_seen

    writer.add_scalar("Val/loss",        avg_loss,  epoch)
    writer.add_scalar("Val/top1",    100 * top1,    epoch)
    writer.add_scalar("Val/top5",    100 * top5,    epoch)
    writer.add_scalar("Val/sibling", 100 * sib,     epoch)
    print(f"Val   — loss: {avg_loss:.4f}, top1: {100*top1:.1f}%, top5: {100*top5:.1f}%, sibling: {100*sib:.1f}%")
    return {"val_loss": avg_loss, "val_top1": top1, "val_top5": top5, "val_sibling": sib}


Path(os.path.join(BASE_DIR, "checkpoints/fgvc", params.name)).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(os.path.join(BASE_DIR, "convnext_fgvc", params.name))

best_top1     = -1.0
best_top5     = -1.0
best_val_loss = float("inf")
start_epoch   = 0
checkpoint_path = os.path.join(BASE_DIR, "checkpoints/fgvc", params.name, "checkpoint.pth")

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["lr_scheduler"])
    start_epoch = ckpt["epoch"] + 1

    if "metrics" in ckpt: #restore best metrics so we can save future best ones correctly
        best_top1 = ckpt["metrics"]["val_top1"]
        best_val_loss = ckpt["metrics"]["val_loss"]
        best_top5 = ckpt["metrics"]["val_top5"]

for epoch in range(start_epoch, params.epochs):
    print(f"\nEpoch {epoch+1}/{params.epochs}")
    train_metrics = train(train_loader, model, loss_fn, optimizer, epoch, writer)
    scheduler.step()
    val_metrics = test(test_loader, model, loss_fn, epoch, writer)

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints/fgvc", params.name)
    ckpt = {
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "params":       params.__dict__,
        "metrics":      {**train_metrics, **val_metrics},
    }
    torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pth"))

    if val_metrics["val_top1"] > best_top1:
        best_top1 = val_metrics["val_top1"]
        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best_top1.pth"))

    if val_metrics["val_top5"] > best_top5:
        best_top5 = val_metrics["val_top5"]
        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best_top5.pth"))

    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best_loss.pth"))