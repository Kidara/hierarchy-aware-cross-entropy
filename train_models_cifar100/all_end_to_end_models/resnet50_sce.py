import numpy as np
import pickle
import time
import os
from pathlib import Path
from CIFAR100_class import CIFAR100Dataset

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, TensorDataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from data_preprocessing import fine_to_siblings

BASE_DIR = "/data/user"
os.makedirs(BASE_DIR, exist_ok=True)

#set parameters for training
class Params:
    def __init__(self):
        self.batch_size = 128
        self.name = "resnet50_sce_no_smooth"
        self.workers = 8
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.num_classes = 100
        self.epochs = 450
        self.warmup_epochs = 10
        self.label_smoothing = 0

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

#specify device
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_scheduler, initial_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.initial_lr = initial_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.base_scheduler.step()
        self.current_epoch += 1
    
    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'base_scheduler': self.base_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])

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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761])
])

#train dataset (50k) - hce=False for standard cross-entropy
train_dataset = CIFAR100Dataset(train_data, transform=train_transform, hce=False)
#CIFAR-100 test set (10k) for reporting
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

def train(dataloader, model, loss_fn, optimizer, scaler, epoch, writer, approx_size=50000):
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
        X, y_fine = X.to(device), y_fine.to(device)  #using fine labels
        y = y_fine
        
        optimizer.zero_grad()
        
        #mixed precision training
        with torch.cuda.amp.autocast():
            pred = model(X)
            loss = loss_fn(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
            
            pred_top1 = pred.argmax(1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
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
    top1 = correct
    top5 = correct_top5
    correct_sibling /= samples_seen

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

# ResNet50 model adapted for CIFAR-100
model = models.resnet50(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, params.num_classes)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=params.lr,
                            momentum=params.momentum,
                            weight_decay=params.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=params.epochs,
    eta_min=0
)

#warmup scheduler
warmup_scheduler = WarmupScheduler(optimizer, params.warmup_epochs, scheduler, params.lr)
#mixed precision scaler
scaler = torch.cuda.amp.GradScaler()

#create directory on disk to save model checkpoints
Path(os.path.join(BASE_DIR, "checkpoints/cifar100", params.name)).mkdir(parents=True, exist_ok=True)
#create TensorBoard logger that writes training statistics
writer = SummaryWriter(os.path.join(BASE_DIR, "resnet50_cifar100", params.name))

start_epoch = 0 #used if we need to restore from a checkpoint
num_epochs = params.epochs
best_top1 = -1.0 #use this variable to check best top1 accuracy so far and save that model weights
best_top5 = -1.0
best_val_loss = float("inf") #use this variable to checkpoint model with least validation loss
checkpoint_path = os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint.pth") #check if checkpoint_path exists so we restore where training left off

# if os.path.exists(checkpoint_path):
#     print(f"Loading checkpoint...")
#     ckpt = torch.load(checkpoint_path)
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     warmup_scheduler.load_state_dict(ckpt["lr_scheduler"])
#     scaler.load_state_dict(ckpt["scaler"])
#     start_epoch = ckpt["epoch"] + 1
#
#     if "metrics" in ckpt: #restore best metrics so we can save future best ones correctly
#         best_top1 = ckpt["metrics"]["val_top1"]
#         best_val_loss = ckpt["metrics"]["val_loss"]
#         best_top5 = ckpt["metrics"]["val_top5"]


for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    train_metrics = train(train_loader, model, criterion, optimizer, scaler, epoch=epoch, writer=writer)
    warmup_scheduler.step()
    val_metrics = test(test_loader, model, criterion, epoch=epoch, writer=writer)
    
    # Log learning rate
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    #checkpoint
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": warmup_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
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