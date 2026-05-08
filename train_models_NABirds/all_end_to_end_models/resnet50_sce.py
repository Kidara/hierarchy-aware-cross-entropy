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

BASE_DIR = "/data/user"
NAB_DIR = os.path.join(BASE_DIR, "nabirds")
os.makedirs(BASE_DIR, exist_ok=True)

class Params:
    def __init__(self):
        self.batch_size = 128
        self.name = "resnet50_sce_smooth"
        self.workers = 8
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.num_classes = 555
        self.label_smoothing = 0.1
        self.epochs = 750

    def __repr__(self):
        return str(self.__dict__)

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

params = Params()
raw_labels = {}
with open(os.path.join(NAB_DIR, "image_class_labels.txt")) as f:
    for line in f:
        img_id, label = line.strip().split()
        raw_labels[img_id] = int(label)

all_taxonomy_ids = sorted(set(raw_labels.values()))  #555 unique node IDs
taxonomy_to_seq = {tax_id: i for i, tax_id in enumerate(all_taxonomy_ids)}
seq_to_taxonomy = {i: tax_id for tax_id, i in taxonomy_to_seq.items()}

#remap sibling matrix from 1011-dim taxonomy space to 555-dim sequential space
sibling_matrix_full = pd.read_csv('sibling_matrix.csv', index_col=0)
sibling_matrix_full.index = sibling_matrix_full.index.astype(int)
sibling_matrix_full.columns = sibling_matrix_full.columns.astype(int)

S_small = np.zeros((555, 555), dtype=np.float32)
for seq_i, tax_i in seq_to_taxonomy.items():
    for seq_j, tax_j in seq_to_taxonomy.items():
        S_small[seq_i, seq_j] = sibling_matrix_full.loc[tax_i, tax_j]

S = torch.from_numpy(S_small).to(device)

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
                image_labels[img_id] = taxonomy_to_seq[int(label)]

        is_train = (split == "train")
        self.samples = []
        for img_id, flag in split_map.items():
            if bool(flag) == is_train:
                path = os.path.join(nabirds_dir, "images", image_paths[img_id])
                self.samples.append((path, image_labels[img_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

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

train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                          shuffle=True,  num_workers=params.workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=params.batch_size,
                          shuffle=False, num_workers=params.workers, pin_memory=True)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, params.num_classes)
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing = 0.1)

optimizer = torch.optim.SGD(model.parameters(), lr=params.lr,
                            momentum=params.momentum, weight_decay=params.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)

def sibling_accuracy(pred_top1, targets):
    correct = (pred_top1 == targets)
    is_sibling = S[targets, pred_top1].bool()
    return (correct | is_sibling).sum().item()

def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    correct_sibling = 0
    samples_seen = 0
    num_batches = 0
    start = time.time()

    for batch, (X, labels) in enumerate(dataloader):
        X = X.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        samples_seen += len(X)
        running_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            pred_top1 = pred.argmax(1)
            correct_top1 += (pred_top1 == labels).sum().item()
            _, pred_top5 = pred.topk(5, dim=1)
            correct_top5 += pred_top5.eq(labels.unsqueeze(1)).sum().item()
            correct_sibling += sibling_accuracy(pred_top1, labels)

        if batch % 100 == 0:
            print(f"  batch {batch}, loss: {loss.item():.4f}, elapsed: {time.time()-start:.1f}s")

    avg_loss = running_loss / num_batches
    top1 = correct_top1 / samples_seen
    top5 = correct_top5 / samples_seen
    sib = correct_sibling / samples_seen

    writer.add_scalar("Train/loss", avg_loss, epoch)
    writer.add_scalar("Train/top1", 100 * top1, epoch)
    writer.add_scalar("Train/top5", 100 * top5, epoch)
    writer.add_scalar("Train/sibling_acc", 100 * sib, epoch)
    print(f"Train — loss: {avg_loss:.4f}, top1: {100*top1:.1f}%, top5: {100*top5:.1f}%, sibling: {100*sib:.1f}%")
    return {"train_loss": avg_loss, "train_top1": top1, "train_top5": top5, "train_sibling": sib}


def test(dataloader, model, loss_fn, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    correct_sibling = 0
    samples_seen = 0
    num_batches = 0

    with torch.no_grad():
        for X, labels in dataloader:
            X = X.to(device)
            labels = labels.to(device)

            pred = model(X)
            running_loss += loss_fn(pred, labels).item()
            num_batches += 1
            samples_seen += len(X)

            pred_top1 = pred.argmax(1)
            correct_top1 += (pred_top1 == labels).sum().item()
            _, pred_top5 = pred.topk(5, dim=1)
            correct_top5 += pred_top5.eq(labels.unsqueeze(1)).sum().item()
            correct_sibling += sibling_accuracy(pred_top1, labels)

    avg_loss = running_loss / num_batches
    top1 = correct_top1 / samples_seen
    top5 = correct_top5 / samples_seen
    sib = correct_sibling / samples_seen

    writer.add_scalar("Val/loss", avg_loss, epoch)
    writer.add_scalar("Val/top1", 100 * top1, epoch)
    writer.add_scalar("Val/top5", 100 * top5, epoch)
    writer.add_scalar("Val/sibling_acc", 100 * sib, epoch)
    print(f"Val   — loss: {avg_loss:.4f}, top1: {100*top1:.1f}%, top5: {100*top5:.1f}%, sibling: {100*sib:.1f}%")
    return {"val_loss": avg_loss, "val_top1": top1, "val_top5": top5, "val_sibling": sib}


Path(os.path.join(BASE_DIR, "checkpoints/NA_Birds", params.name)).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(os.path.join(BASE_DIR, "resnet50_NABirds", params.name))

best_top1 = -1.0
best_top5 = -1.0
best_val_loss = float("inf")
start_epoch = 0
# checkpoint_path = os.path.join(BASE_DIR, "checkpoints", params.name, "checkpoint.pth") #check if checkpoint_path exists so we restore where training left off
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

for epoch in range(start_epoch, params.epochs):
    print(f"\nEpoch {epoch+1}/{params.epochs}")
    train_metrics = train(train_loader, model, loss_fn, optimizer, epoch, writer)
    scheduler.step()
    val_metrics = test(test_loader, model, loss_fn, epoch, writer)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "params": params.__dict__,
        "metrics": {**train_metrics, **val_metrics},
    }
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints/NA_Birds", params.name)
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