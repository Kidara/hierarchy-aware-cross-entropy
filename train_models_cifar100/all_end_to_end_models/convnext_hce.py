#train model without R.T and just R
#go back to R.T and don't use diluted encoding (just 1 everywhere)
import numpy as np
import pickle
import time
import os
from pathlib import Path
from CIFAR100_class import CIFAR100Dataset
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, TensorDataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

BASE_DIR = "/data/user"
os.makedirs(BASE_DIR, exist_ok=True)


# hyperparameters recipe: https://arxiv.org/pdf/2201.03545

class Params:
    def __init__(self):
        self.batch_size    = 128
        self.name          = "convnext_hce_0.7_no_smooth_adj"
        self.workers       = 8
        self.lr            = 4e-3 * (128 / 4096) * (10 / 7)
        self.momentum      = 0.9
        self.weight_decay  =  0.05
        self.num_classes   = 120
        self.label_smoothing = 0
        self.epochs          = 400

    def __repr__(self):
        return str(self.__dict__)

#specify device
device = (
    "cuda:5"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

hierarchy_matrix  = np.load('hierarchy_matrix.npy')
diluted_encoding  = np.load('diluted_encoding.npy')
with open('fine_to_siblings.pkl', 'rb') as f:
    fine_to_siblings = pickle.load(f)

#read train/val data 
train_data = unpickle('cifar-100-python/train')
test_data = unpickle('cifar-100-python/test')

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761])
])

#create train dataset
train_dataset = CIFAR100Dataset(train_data, transform=train_transform, hce=True)
#create test dataset
test_dataset = CIFAR100Dataset(test_data, transform=test_transform, hce=True)

params = Params()

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
        #for accuracy, just use last 100 (sure that we are always picking a leaf)
        with torch.no_grad():
            pred_top1 = pred.argmax(1)
            pred_top1_leaves = torch.where(pred_top1 >= 20, pred_top1, torch.tensor(-1, device=device))
            correct_top1 += (pred_top1_leaves == y).sum().item()
            
            _, pred_top5 = pred.topk(5, dim=1)
            pred_top5_leaves = torch.where(pred_top5 >= 20, pred_top5, torch.tensor(-1, device=device))
            correct_top5 += pred_top5_leaves.eq(y.view(-1, 1)).sum().item()
            
            for i in range(len(y)):
                true_label = y[i].item() - 20 #account for offset because we mapped this using 0 - 100
                predicted_label = pred_top1[i].item() - 20
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
            pred_top1_leaves = torch.where(pred_top1 >= 20, pred_top1, torch.tensor(-1, device=device))
            correct += (pred_top1_leaves == y).type(torch.float).sum().item()
            
            _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
            pred_top5_leaves = torch.where(pred_top5 >= 20, pred_top5, torch.tensor(-1, device=device))
            correct_top5 += pred_top5_leaves.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

            for i in range(len(y)):
                true_label = y[i].item() - 20
                predicted_label = pred_top1[i].item() - 20
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

model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, params.num_classes)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)

#create reachability matrix
R_node_leaf = hierarchy_matrix
T_node_leaf = diluted_encoding
#convert to torch tensors
R = torch.from_numpy(R_node_leaf.astype(np.float32)).to(device)
T = torch.from_numpy(T_node_leaf.astype(np.float32)).to(device)
print(f"R tensor shape: {R.shape}")
print(f"T tensor shape: {T.shape}")

#hce loss function
def hierarchical_cross_entropy_loss(logits, targets, alpha = params.label_smoothing):
    #logits size is [batch, num classes]
    cell_type_probs = torch.softmax(logits, dim=-1)
    cell_type_probs = torch.matmul(cell_type_probs, R.T)
    cell_type_probs = torch.log(cell_type_probs + 1e-6)

    #smooth labels
    batch_size = targets.size(0)
    smoothed_labels = torch.full((batch_size, 100), alpha / (99), device=logits.device)
    original_fine_labels = targets - 20
    smoothed_labels.scatter_(1, original_fine_labels.unsqueeze(1), 1 - alpha)

    y_node = torch.matmul(smoothed_labels, T.T)
    loss = -(y_node * cell_type_probs).sum(dim=1).mean()
    return loss

#create directory on disk to save model checkpoints
Path(os.path.join(BASE_DIR, "checkpoints/cifar100", params.name)).mkdir(parents=True, exist_ok=True)
#create TensorBoard logger that writes training statistics
writer = SummaryWriter(os.path.join(BASE_DIR, "convnext_cifar100", params.name))

start_epoch = 0 #used if we need to restore from a checkpoint
num_epochs = params.epochs
best_top1 = -1.0 #use this variable to check best top1 accuracy so far and save that model weights
best_top5 = -1.0
best_val_loss = float("inf") #use this variable to checkpoint model with least validation loss

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

for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_metrics = train(train_loader, model, hierarchical_cross_entropy_loss, optimizer, epoch=epoch, writer=writer)
    scheduler.step()
    val_metrics = test(test_loader, model, hierarchical_cross_entropy_loss, epoch=epoch, writer=writer)

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
    
    if val_metrics["val_top5"] > best_top5:
        best_top5 = val_metrics["val_top5"]
        torch.save(ckpt, os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint_best_top5accuracy.pth"))
    
    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        torch.save(ckpt, os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint_best_loss.pth"))