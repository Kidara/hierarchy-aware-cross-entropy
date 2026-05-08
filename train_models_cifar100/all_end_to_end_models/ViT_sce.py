import numpy as np
import pickle
import time
import os
from pathlib import Path
from CIFAR100_class import CIFAR100Dataset
# from data_preprocessing import fine_to_siblings

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.vision_transformer import VisionTransformer

BASE_DIR = "/data/user"
os.makedirs(BASE_DIR, exist_ok=True)

hierarchy_matrix  = np.load('hierarchy_matrix.npy')
diluted_encoding  = np.load('diluted_encoding.npy')
with open('fine_to_siblings.pkl', 'rb') as f:
    fine_to_siblings = pickle.load(f)

class Params:
    def __init__(self):
        self.batch_size = 64
        self.name = "vit_sce_0.7_no_smooth_adj2"
        self.workers = 8
        self.lr = 3e-3 / (10/7)
        self.weight_decay = 0.3
        self.dropout = 0.1
        self.num_classes = 100
        self.epochs = 300
        self.label_smoothing = 0

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
#specify device
device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Net(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=12,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=1024,
            dropout=0.1,
            attention_dropout=0.1,
            num_classes=num_classes
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

params = Params()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#read train/val data 
train_data = unpickle('cifar-100-python/train')
test_data = unpickle('cifar-100-python/test')

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761])
])

#create train dataset and split in train and validation (85/15)
train_dataset = CIFAR100Dataset(train_data, transform=train_transform, hce=False)
#create test dataset
test_dataset = CIFAR100Dataset(test_data, transform=test_transform, hce=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=params.batch_size,
    shuffle=True,
    num_workers=params.workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=params.batch_size,
    shuffle=False,
    num_workers=params.workers,
    pin_memory=True
)

def train(dataloader, model, loss_fn, optimizer, epoch, writer, approx_size=42500):
    model.train()
    start0 = time.time()
    start = time.time()
    running_loss = 0.0
    samples_seen = 0
    num_batches = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_sibling = 0
    
    for batch, (X, y_fine, y_coarse) in enumerate(dataloader):
        num_batches += 1
        X, y_fine = X.to(device), y_fine.to(device)
        y = y_fine
        
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        
        batch_size = len(X)
        samples_seen += batch_size
        running_loss += loss.item()

        #train accuracy
        with torch.no_grad():
            pred_top1 = pred.argmax(1)
            correct_top1 += (pred_top1 == y).sum().item()
            
            _, pred_top5 = pred.topk(5, dim=1)
            correct_top5 += pred_top5.eq(y.view(-1, 1)).sum().item()
            
            for i in range(len(y)):
                true_label = y[i].item()
                predicted_label = pred_top1[i].item()
                if predicted_label in fine_to_siblings[true_label]:
                    correct_sibling += 1
        
        if batch % 100 == 0:
            current = samples_seen
            print(f"loss: {loss.item():>7f}  [{current:>7d}/{approx_size:>7d}], {(current/approx_size * 100):>5.2f}%")
            
            new_start = time.time()
            delta = new_start - start
            start = new_start
            
            if batch != 0:
                print(f"Last 100 batches done in {delta:.2f} seconds")
                remaining_samples = approx_size - current
                speed = 100 * batch_size / delta
                remaining_time = remaining_samples / speed
                print(f"Estimated remaining time: {remaining_time:.2f} seconds ({remaining_time/60:.2f} minutes)")
    
    print(f"Entire epoch done in {time.time() - start0:.2f} seconds")
    avg_train_loss = running_loss / num_batches
    print(f"Train avg loss: {avg_train_loss:.6f}")

    train_top1 = correct_top1 / samples_seen
    train_top5 = correct_top5 / samples_seen
    correct_sibling /= samples_seen
    writer.add_scalar("Train/loss", avg_train_loss, epoch)
    writer.add_scalar("Train/top1", 100 * train_top1, epoch)
    writer.add_scalar("Train/top5", 100 * train_top5, epoch)
    writer.add_scalar('Train/sibling_acc', 100*correct_sibling, epoch)

    return {"train_loss": avg_train_loss, "train_top1": train_top1, "train_top5": train_top5, "train_sibling_acc": correct_sibling}

#test function that computes top-5 accuracy and top-1 accuracy
#iterates through entire validation data and reports them to the console and tensorboard
def test(dataloader, model, loss_fn, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0
    num_batches = 0
    samples_seen = 0
    correct_sibling = 0
    
    with torch.no_grad():
        for X, y_fine, y_coarse in dataloader:
            X, y_fine = X.to(device), y_fine.to(device)
            y = y_fine

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            pred_top1 = pred.argmax(1)
            correct += (pred_top1 == y).type(torch.float).sum().item()
            
            _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

            for i in range(len(y)):
                true_label = y[i].item()
                predicted_label = pred_top1[i].item()
                if predicted_label in fine_to_siblings[true_label]:
                    correct_sibling += 1

            num_batches += 1
            samples_seen += len(X)
    
    test_loss /= num_batches
    correct /= samples_seen
    correct_top5 /= samples_seen
    correct_sibling /= samples_seen
    top1 = correct
    top5 = correct_top5

    writer.add_scalar('Val/loss', test_loss, epoch)
    writer.add_scalar('Val/top1', 100*top1, epoch)
    writer.add_scalar('Val/top5', 100*top5, epoch)
    writer.add_scalar('Val/sibling_acc', 100*correct_sibling, epoch)
    
    print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Top-5 Accuracy: {(100*correct_top5):>0.1f}%\n")
    
    return {
        "val_loss": test_loss,
        "val_top1": top1,
        "val_top5": top5,
        "val_sibling_acc": correct_sibling,
    }

model = models.vit_b_16(weights=None, dropout=params.dropout)
model.heads = nn.Linear(model.heads.head.in_features, params.num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)

#AdamW with weight_decay=0.3 per ViT-* ImageNet recipe (Dosovitskiy et al.)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=params.lr,
    weight_decay=params.weight_decay,
    betas=(0.9, 0.999),
)

# Cosine decay with linear warmup
warmup_epochs = 10
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(max(1, params.epochs - warmup_epochs))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#create directory on disk to save model checkpoints
Path(os.path.join(BASE_DIR, "checkpoints/cifar100", params.name)).mkdir(parents=True, exist_ok=True)
#create TensorBoard logger that writes training statistics
writer = SummaryWriter(os.path.join(BASE_DIR, "vit_cifar100", params.name))

start_epoch = 0
num_epochs = params.epochs
best_top1 = -1.0
best_top5 = -1.0
best_val_loss = float("inf")
checkpoint_path = os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint.pth")

# Uncomment to resume training
# if os.path.exists(checkpoint_path):
#     print(f"Loading checkpoint...")
#     ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
#     scheduler.load_state_dict(ckpt["lr_scheduler"])
#     start_epoch = ckpt["epoch"] + 1

#     if "metrics" in ckpt:
#         best_top1 = ckpt["metrics"]["val_top1"]
#         best_val_loss = ckpt["metrics"]["val_loss"]
#         best_top5 = ckpt["metrics"]["val_top5"]

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    train_metrics = train(train_loader, model, criterion, optimizer, epoch=epoch, writer=writer)
    scheduler.step()
    val_metrics = test(test_loader, model, criterion, epoch=epoch, writer=writer)
    
    # Log learning rate
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    #checkpoint
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "params": params.__dict__,
        "metrics": {
            "train_loss": train_metrics["train_loss"],
            "train_top1": train_metrics["train_top1"],
            "train_top5": train_metrics["train_top5"],
            "val_loss": val_metrics["val_loss"],
            "val_top1": val_metrics["val_top1"],
            "val_top5": val_metrics["val_top5"],
        }
    }
    torch.save(ckpt, os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint.pth"))
    
    if val_metrics["val_top1"] > best_top1:
        best_top1 = val_metrics["val_top1"]
        torch.save(ckpt, os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint_best_accuracy.pth"))
        print(f"New best top-1 accuracy: {100*best_top1:.2f}%")
    
    if val_metrics["val_top5"] > best_top5:
        best_top5 = val_metrics["val_top5"]
        torch.save(ckpt, os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint_best_top5accuracy.pth"))
        print(f"New best top-5 accuracy: {100*best_top5:.2f}%")
    
    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        torch.save(ckpt, os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint_best_loss.pth"))
        print(f"New best validation loss: {best_val_loss:.6f}")

print("Training completed!")
print(f"Best Top-1 Accuracy: {100*best_top1:.2f}%")
print(f"Best Top-5 Accuracy: {100*best_top5:.2f}%")
print(f"Best Validation Loss: {best_val_loss:.6f}")
