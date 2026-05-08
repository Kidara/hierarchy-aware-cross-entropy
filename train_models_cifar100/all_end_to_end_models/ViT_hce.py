import numpy as np
import pickle
import time
import os
from pathlib import Path
from CIFAR100_class import CIFAR100Dataset
import torchvision.models as models
import matplotlib.pyplot as plt
# from data_preprocessing import hierarchy_matrix, diluted_encoding, fine_to_siblings

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.vision_transformer import VisionTransformer

BASE_DIR = "/data/user"
os.makedirs(BASE_DIR, exist_ok=True)

class Params:
    def __init__(self):
        self.batch_size = 64
        self.name = "vit_hce_0.7_no_smooth_adj2"
        self.workers = 8
        self.lr = 3e-3 * (10/7)
        self.weight_decay = 0.3
        self.dropout = 0.1
        self.num_classes = 120
        self.epochs = 300
        self.label_smoothing = 0

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

#specify device
device = (
    "cuda:4"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
    
class Net(nn.Module):
    def __init__(self, num_classes=120):
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

hierarchy_matrix  = np.load('hierarchy_matrix.npy')
diluted_encoding  = np.load('diluted_encoding.npy')
with open('fine_to_siblings.pkl', 'rb') as f:
    fine_to_siblings = pickle.load(f)

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

# model = Net(params.num_classes)
# model = model.to(device)
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=params.lr,
#     weight_decay=params.weight_decay
# )
model = models.vit_b_16(weights=None, dropout=params.dropout)
model.heads = nn.Linear(model.heads.head.in_features, params.num_classes)
model = model.to(device)

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
writer = SummaryWriter(os.path.join(BASE_DIR, "vit_cifar100", params.name))

start_epoch = 0 #used if we need to restore from a checkpoint
num_epochs = params.epochs
best_top1 = -1.0 #use this variable to check best top1 accuracy so far and save that model weights
best_top5 = -1.0
best_val_loss = float("inf") #use this variable to checkpoint model with least validation loss
checkpoint_path = os.path.join(BASE_DIR, "checkpoints/cifar100", params.name, "checkpoint.pth") #check if checkpoint_path exists so we restore where training left off

# if os.path.exists(checkpoint_path):
#     print(f"Loading checkpoint...")
#     ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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

#best accuracy paths
# checkpoint_path_hce = os.path.join(BASE_DIR, "checkpoints", "vit_hce_0.2_no_smooth", "checkpoint_best_accuracy.pth") #check if checkpoint_path exists so we restore where training left off
# checkpoint_path_sce = os.path.join(BASE_DIR, "checkpoints", "vit_sce_no_smooth", "checkpoint_best_accuracy.pth")

# # load fine label names for printing
# meta_data = unpickle('cifar-100-python/meta')
# fine_label_names = [label.decode('utf-8') for label in meta_data[b'fine_label_names']]

# model_hce = Net(120)
# model_hce = model_hce.to(device)

# model_sce = Net(100)
# model_sce = model_sce.to(device)

# meta_data = unpickle("cifar-100-python/meta")
# fine_label_names = [label.decode("utf-8") for label in meta_data[b"fine_label_names"]]

# def load_ckpt(path):
#     try:
#         return torch.load(path, map_location=device)
#     except Exception:
#         return torch.load(path, map_location=device, weights_only=False)

# ckpt_hce = load_ckpt(checkpoint_path_hce)
# model_hce.load_state_dict(ckpt_hce["model"])

# ckpt_sce = load_ckpt(checkpoint_path_sce)
# model_sce.load_state_dict(ckpt_sce["model"])

# def per_class_top1_top5_and_sibling_both(dataloader, model_hce, model_sce, k=5):
#     model_hce.eval()
#     model_sce.eval()

#     total = np.zeros(100, dtype=np.int64)
#     correct1_hce = np.zeros(100, dtype=np.int64)
#     sib1_hce     = np.zeros(100, dtype=np.int64)
#     correct1_sce = np.zeros(100, dtype=np.int64)
#     sib1_sce     = np.zeros(100, dtype=np.int64)
#     correctk_hce = np.zeros(100, dtype=np.int64)
#     sibk_hce     = np.zeros(100, dtype=np.int64)
#     correctk_sce = np.zeros(100, dtype=np.int64)
#     sibk_sce     = np.zeros(100, dtype=np.int64)

#     with torch.no_grad():
#         for X, y_fine, y_coarse in dataloader:
#             X = X.to(device)
#             y_fine = y_fine.to(device)
#             y_true = (y_fine - 20).detach().cpu().numpy().astype(int)
#             logits_hce = model_hce(X)                                  
#             pred1_hce = logits_hce.argmax(1).detach().cpu().numpy()
#             pred1_hce = np.where(pred1_hce >= 20, pred1_hce - 20, -1)
#             topk_hce = torch.topk(logits_hce, k=k, dim=1).indices.detach().cpu().numpy()
#             topk_hce = np.where(topk_hce >= 20, topk_hce - 20, -1)

#             logits_sce = model_sce(X)                                 
#             pred1_sce  = logits_sce.argmax(1).detach().cpu().numpy()
#             topk_sce   = torch.topk(logits_sce, k=k, dim=1).indices.detach().cpu().numpy()

#             for i in range(len(y_true)):
#                 t = int(y_true[i])
#                 total[t] += 1

#                 ph1 = int(pred1_hce[i])
#                 ps1 = int(pred1_sce[i])

#                 if ph1 == t:
#                     correct1_hce[t] += 1
#                     sib1_hce[t] += 1
#                 else:
#                     if ph1 in fine_to_siblings[t]:
#                         sib1_hce[t] += 1

#                 if ps1 == t:
#                     correct1_sce[t] += 1
#                     sib1_sce[t] += 1
#                 else:
#                     if ps1 in fine_to_siblings[t]:
#                         sib1_sce[t] += 1

#                 hset = set(topk_hce[i].tolist())
#                 sset = set(topk_sce[i].tolist())

#                 if t in hset:
#                     correctk_hce[t] += 1
#                     sibk_hce[t] += 1
#                 else:
#                     if any(p in fine_to_siblings[t] for p in hset):
#                         sibk_hce[t] += 1

#                 if t in sset:
#                     correctk_sce[t] += 1
#                     sibk_sce[t] += 1
#                 else:
#                     if any(p in fine_to_siblings[t] for p in sset):
#                         sibk_sce[t] += 1

#     def safe_div(num):
#         return np.divide(num, total, out=np.zeros_like(num, dtype=float), where=total > 0)

#     acc1_hce = safe_div(correct1_hce)
#     sib1_hce = safe_div(sib1_hce)
#     acc1_sce = safe_div(correct1_sce)
#     sib1_sce = safe_div(sib1_sce)

#     acck_hce = safe_div(correctk_hce)
#     sibk_hce = safe_div(sibk_hce)
#     acck_sce = safe_div(correctk_sce)
#     sibk_sce = safe_div(sibk_sce)

#     return acc1_hce, sib1_hce, acc1_sce, sib1_sce, acck_hce, sibk_hce, acck_sce, sibk_sce, total


# acc1_hce, sib1_hce, acc1_sce, sib1_sce, acc5_hce, sib5_hce, acc5_sce, sib5_sce, total = \
#     per_class_top1_top5_and_sibling_both(test_loader, model_hce, model_sce, k=5)

# def plot_per_class_delta(
#     acc_hce, acc_sce, fine_label_names, out_path, title, sort_by="sce"
# ):
#     hce_plot = 100 * acc_hce
#     sce_plot = 100 * acc_sce

#     if sort_by == "sce":
#         order = np.argsort(-sce_plot)
#     elif sort_by == "hce":
#         order = np.argsort(-hce_plot)
#     else:
#         raise ValueError("sort_by must be 'sce' or 'hce'")

#     sce_sorted = sce_plot[order]
#     hce_sorted = hce_plot[order]
#     labels = [fine_label_names[i] for i in order]

#     diff = hce_sorted - sce_sorted
#     x = np.arange(len(sce_sorted))

#     plt.figure(figsize=(28, 8))
#     plt.bar(x, sce_sorted, color="lightgrey", width=0.9)
#     pos = diff > 0
#     neg = diff < 0
#     plt.bar(x[pos], diff[pos], bottom=sce_sorted[pos], color="green", width=0.9)
#     plt.bar(x[neg], -diff[neg], bottom=hce_sorted[neg], color="red", width=0.9)

#     plt.ylim(0, 100)
#     plt.xlabel(f"Class (sorted by {sort_by.upper()} accuracy)")
#     plt.ylabel("Accuracy (%)")
#     plt.title(title)
#     plt.xticks(x, labels, rotation=90, fontsize=7)

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     print(f"Saved plot: {out_path}")

# plot_path_top1 = os.path.join(BASE_DIR, "resnet18_hce_vs_sce_top1_per_class.png")
# plot_path_top5 = os.path.join(BASE_DIR, "resnet18_hce_vs_sce_top5_per_class.png")

# plot_per_class_delta(
#     acc_hce=acc1_hce,
#     acc_sce=acc1_sce,
#     fine_label_names=fine_label_names,
#     out_path=plot_path_top1,
#     title="ViT Per-class TOP-1 accuracy: SCE (not adjusted/no smooth) baseline vs. HCE (not adjusted/no smooth)",
#     sort_by="sce",
# )

# plot_per_class_delta(
#     acc_hce=acc5_hce,
#     acc_sce=acc5_sce,
#     fine_label_names=fine_label_names,
#     out_path=plot_path_top5,
#     title="ViT Per-class TOP-5 accuracy: SCE (not adjusted/no smooth) baseline vs. HCE (not adjusted/no smooth)",
#     sort_by="sce",
# )

# print(f"HCE mean top1: {100 * acc1_hce.mean():.2f}%")
# print(f"SCE mean top1: {100 * acc1_sce.mean():.2f}%")
# print(f"HCE mean top5: {100 * acc5_hce.mean():.2f}%")
# print(f"SCE mean top5: {100 * acc5_sce.mean():.2f}%")

# ckpt_hce = torch.load(checkpoint_path_hce, map_location=device, weights_only=False)
# model_hce.load_state_dict(ckpt_hce["model"])

# ckpt_sce = torch.load(checkpoint_path_sce, map_location=device, weights_only=False)
# model_sce.load_state_dict(ckpt_sce["model"])

# def per_class_accuracy_and_sibling_accuracy_both(dataloader, model_hce, model_sce):
#     model_hce.eval()
#     model_sce.eval()
#     total = np.zeros(100, dtype=np.int64)

#     correct_hce = np.zeros(100, dtype=np.int64)
#     sibling_correct_hce = np.zeros(100, dtype=np.int64)

#     correct_sce = np.zeros(100, dtype=np.int64)
#     sibling_correct_sce = np.zeros(100, dtype=np.int64)

#     with torch.no_grad():
#         for X, y_fine, y_coarse in dataloader:
#             X, y_fine = X.to(device), y_fine.to(device)
#             y_true = (y_fine - 20).detach().cpu().numpy()

#             logits_hce = model_hce(X)
#             probs_node = torch.softmax(logits_hce, dim=-1)
#             probs_leaf = torch.matmul(probs_node, R.T)[:, 20:120]
#             pred_hce = probs_leaf.argmax(1).detach().cpu().numpy()

#             logits_sce = model_sce(X)
#             pred_sce = logits_sce.argmax(1).detach().cpu().numpy()

#             for i in range(len(y_true)):
#                 t = int(y_true[i])
#                 ph = int(pred_hce[i])
#                 ps = int(pred_sce[i])

#                 total[t] += 1

#                 if ph == t:
#                     correct_hce[t] += 1
#                     sibling_correct_hce[t] += 1
#                 else:
#                     if ph in fine_to_siblings[t]:
#                         sibling_correct_hce[t] += 1

#                 if ps == t:
#                     correct_sce[t] += 1
#                     sibling_correct_sce[t] += 1
#                 else:
#                     if ps in fine_to_siblings[t]:
#                         sibling_correct_sce[t] += 1

#     acc_hce = np.divide(correct_hce, total, out=np.zeros_like(correct_hce, dtype=float), where=total > 0)
#     sib_acc_hce = np.divide(sibling_correct_hce, total, out=np.zeros_like(sibling_correct_hce, dtype=float), where=total > 0)

#     acc_sce = np.divide(correct_sce, total, out=np.zeros_like(correct_sce, dtype=float), where=total > 0)
#     sib_acc_sce = np.divide(sibling_correct_sce, total, out=np.zeros_like(sibling_correct_sce, dtype=float), where=total > 0)

#     return acc_hce, sib_acc_hce, acc_sce, sib_acc_sce, total

# acc_hce, sib_acc_hce, acc_sce, sib_acc_sce, total = per_class_accuracy_and_sibling_accuracy_both(test_loader, model_hce, model_sce)

# order = np.argsort(-acc_hce)

# overall_acc_hce = acc_hce @ total / total.sum()
# overall_sib_acc_hce = sib_acc_hce @ total / total.sum()

# overall_acc_sce = acc_sce @ total / total.sum()
# overall_sib_acc_sce = sib_acc_sce @ total / total.sum()

# print("\nOverall accuracy (ViT with Leaf Weight = 0.2):")
# print(f"HCE: {100*overall_acc_hce:.2f}%")
# print(f"SCE: {100*overall_acc_sce:.2f}%")

# print("\nOverall sibling accuracy:")
# print(f"HCE: {100*overall_sib_acc_hce:.2f}%")
# print(f"SCE: {100*overall_sib_acc_sce:.2f}%")

# # print(
# #     f"{'id':>3s}  "
# #     f"{'class':25s}  "
# #     f"{'occurs':>4s}  "
# #     f"{'HCE acc':>8s}  {'HCE sib':>8s}  "
# #     f"{'SCE acc':>8s}  {'SCE sib':>8s}"
# # )

# # for idx in order:
# #     print(
# #         f"{idx:3d}  "
# #         f"{fine_label_names[idx]:25s}  "
# #         f"{total[idx]:4d}  "
# #         f"{100*acc_hce[idx]:8.2f}  {100*sib_acc_hce[idx]:8.2f}  "
# #         f"{100*acc_sce[idx]:8.2f}  {100*sib_acc_sce[idx]:8.2f}"
# #     )

# import numpy as np
# import matplotlib.pyplot as plt
# import os

# sce = 100 * acc_sce
# hce = 100 * acc_hce

# order = np.argsort(-sce)

# sce = sce[order]
# hce = hce[order]
# labels = [fine_label_names[i] for i in order]

# diff = hce - sce
# x = np.arange(len(sce))

# plt.figure(figsize=(28, 8))

# plt.bar(x, sce, color='lightgrey', width=0.9)

# pos = diff > 0
# neg = diff < 0

# plt.bar(x[pos], diff[pos], bottom=sce[pos], color='green', width=0.9)
# plt.bar(x[neg], -diff[neg], bottom=hce[neg], color='red', width=0.9)

# plt.ylim(25, 100)
# plt.xlabel("Class")
# plt.ylabel("Accuracy (%)")
# plt.title("Per-class accuracy:")

# plt.xticks(x, labels, rotation=90, fontsize=7)

# plt.tight_layout()
# plt.savefig(os.path.join(BASE_DIR, "convnext_0.2_analysis.png"), dpi=300)
# plt.close()
