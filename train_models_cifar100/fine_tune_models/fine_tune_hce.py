import os
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from data_preprocessing import hierarchy_matrix, diluted_encoding, fine_to_siblings

BASE_DIR  = "/data/user"
FEAT_DIR  = os.path.join(BASE_DIR, "dinov2_features")
CKPT_ROOT = os.path.join(BASE_DIR, "checkpoints/cifar100")
TB_ROOT   = os.path.join(BASE_DIR, "dinov2_runs_cifar100/")
DATASET_CFG = {
    "cifar100": {
        "num_nodes":       120,
        "num_fine_classes": 100,
        "fine_label_offset": 20,
    },
}

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DINOV2_FEAT_DIM = 1024   # dinov2-large CLS token dimension

class Params:
    def __init__(self, mode: str, dataset: str):
        self.mode       = mode
        self.dataset    = dataset
        self.batch_size = 1024
        self.epochs     = 2000
        self.lr         = 1e-1 * (10/7)
        self.momentum   = 0.9
        self.workers    = 8
        self.weight_decay = 0
        cfg = DATASET_CFG[dataset]
        self.num_nodes        = cfg["num_nodes"]
        self.num_fine_classes = cfg["num_fine_classes"]
        self.fine_label_offset = cfg["fine_label_offset"]
        self.num_classes = self.num_nodes if mode == "hce" else self.num_fine_classes
        self.name = "fine_tune_hce_0.7_smooth_adj"

    def __repr__(self):
        return str(self.__dict__)



def make_hce_loss(R: torch.Tensor, T: torch.Tensor,
                  fine_label_offset: int, label_smoothing: float = 0.1):
    num_fine = T.shape[1]

    def hce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cell_type_probs = torch.softmax(logits, dim=-1)
        cell_type_probs = torch.matmul(cell_type_probs, R.T)
        log_probs = torch.log(cell_type_probs + 1e-6)

        batch_size = targets.size(0)
        fine_idx = (targets - fine_label_offset).unsqueeze(1)

        if label_smoothing > 0:
            smoothed = torch.full((batch_size, num_fine),
                                  label_smoothing / (num_fine - 1),
                                  device=logits.device)
            smoothed.scatter_(1, fine_idx, 1.0 - label_smoothing)
        else:
            smoothed = torch.zeros(batch_size, num_fine, device=logits.device)
            smoothed.scatter_(1, fine_idx, 1.0)

        y_node = torch.matmul(smoothed, T.T)
        loss   = -(y_node * log_probs).sum(dim=1).mean()
        return loss

    return hce_loss


def make_sce_loss(label_smoothing: float = 0.1):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

def load_feature_dataset(dataset: str, split: str):
    path = os.path.join(FEAT_DIR, f"{dataset}_{split}_dinov2.pt")
    assert os.path.exists(path), (
        f"Feature file not found: {path}\n"
        f"Run: python extract_features_dinov2.py --dataset {dataset} --split {split}"
    )
    data = torch.load(path)
    return data["features"], data["fine_labels"]


def make_loaders(params: Params):
    train_feats, train_labels = load_feature_dataset(params.dataset, "train")
    test_feats,  test_labels  = load_feature_dataset(params.dataset, "test")
    if params.mode == "hce":
        train_labels = train_labels + params.fine_label_offset
        test_labels  = test_labels  + params.fine_label_offset

    train_ds = TensorDataset(train_feats, train_labels)
    test_ds  = TensorDataset(test_feats,  test_labels)

    train_loader = DataLoader(train_ds, batch_size=params.batch_size,
                              shuffle=True,  num_workers=params.workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=params.batch_size,
                              shuffle=False, num_workers=params.workers,
                              pin_memory=True)
    return train_loader, test_loader



def _top1_top5_sibling(pred_logits, targets, fine_label_offset, device, is_hce):
    B = len(targets)
    pred_top1 = pred_logits.argmax(1)

    if is_hce:
        pred_top1_leaves = torch.where(
            pred_top1 >= fine_label_offset, pred_top1,
            torch.tensor(-1, device=device)
        )
    else:
        pred_top1_leaves = pred_top1  # all 100 classes are valid

    c1 = (pred_top1_leaves == targets).sum().item()

    _, top5_idx = pred_logits.topk(5, dim=1)
    if is_hce:
        top5_idx = torch.where(top5_idx >= fine_label_offset, top5_idx,
                               torch.tensor(-1, device=device))
    c5 = top5_idx.eq(targets.unsqueeze(1)).sum().item()

    # Normalize to 0-indexed for fine_to_siblings lookup
    offset = fine_label_offset if is_hce else 0
    cs = 0
    for i in range(B):
        true_fine = targets[i].item() - offset
        pred_fine = pred_top1[i].item() - offset
        if pred_fine in fine_to_siblings[true_fine]:
            cs += 1

    return c1, c5, cs


def train(loader, model, loss_fn, optimizer, epoch, writer, params, device):
    model.train()
    t0 = time.time()
    running_loss, n_samples, n_batches = 0.0, 0, 0
    c1_total, c5_total, cs_total = 0, 0, 0

    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(feats)
        loss   = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        B = len(feats)
        n_samples += B
        n_batches += 1
        running_loss += loss.item()

        with torch.no_grad():
            c1, c5, cs = _top1_top5_sibling(
                logits, targets, params.fine_label_offset, device, is_hce=(params.mode == "hce")
            )
        c1_total += c1
        c5_total += c5
        cs_total += cs

    avg_loss = running_loss / n_batches
    top1     = c1_total / n_samples
    top5     = c5_total / n_samples
    sib      = cs_total / n_samples

    writer.add_scalar("Train/loss",        avg_loss,    epoch)
    writer.add_scalar("Train/top1",        100 * top1,  epoch)
    writer.add_scalar("Train/top5",        100 * top5,  epoch)
    writer.add_scalar("Train/sibling_acc", 100 * sib,   epoch)

    print(f"  [train] loss={avg_loss:.4f}  top1={100*top1:.1f}%  "
          f"top5={100*top5:.1f}%  sibling={100*sib:.1f}%  "
          f"({time.time()-t0:.1f}s)")

    return {"train_loss": avg_loss, "train_top1": top1,
            "train_top5": top5,     "train_sibling_acc": sib}


def test(loader, model, loss_fn, epoch, writer, params, device):
    model.eval()
    running_loss, n_samples, n_batches = 0.0, 0, 0
    c1_total, c5_total, cs_total = 0, 0, 0

    with torch.no_grad():
        for feats, targets in loader:
            feats, targets = feats.to(device), targets.to(device)
            logits = model(feats)
            running_loss += loss_fn(logits, targets).item()

            B = len(feats)
            n_samples += B
            n_batches += 1

            c1, c5, cs = _top1_top5_sibling(
                logits, targets, params.fine_label_offset, device, is_hce=(params.mode == "hce")
            )
            c1_total += c1
            c5_total += c5
            cs_total += cs

    avg_loss = running_loss / n_batches
    top1     = c1_total / n_samples
    top5     = c5_total / n_samples
    sib      = cs_total / n_samples

    writer.add_scalar("Val/loss",        avg_loss,    epoch)
    writer.add_scalar("Val/top1",        100 * top1,  epoch)
    writer.add_scalar("Val/top5",        100 * top5,  epoch)
    writer.add_scalar("Val/sibling_acc", 100 * sib,   epoch)

    print(f"  [val]   loss={avg_loss:.4f}  top1={100*top1:.1f}%  "
          f"top5={100*top5:.1f}%  sibling={100*sib:.1f}%")

    return {"val_loss": avg_loss, "val_top1": top1,
            "val_top5": top5,     "val_sibling_acc": sib}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    required=True, choices=["hce", "sce"])
    parser.add_argument("--dataset", default="cifar100",
                        choices=list(DATASET_CFG.keys()))
    parser.add_argument("--gpu",     default="0")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    params = Params(args.mode, args.dataset)
    print(f"Params: {params}")
    train_loader, test_loader = make_loaders(params)
    head = nn.Linear(DINOV2_FEAT_DIM, params.num_classes).to(device)

    if args.mode == "hce":
        R = torch.from_numpy(hierarchy_matrix.astype(np.float32)).to(device)
        T = torch.from_numpy(diluted_encoding.astype(np.float32)).to(device)
        print(f"R: {R.shape}, T: {T.shape}")
        loss_fn = make_hce_loss(R, T,
                                fine_label_offset=params.fine_label_offset,
                                label_smoothing=0.1)
    else:
        loss_fn = make_sce_loss(label_smoothing=0.1)

    optimizer = torch.optim.SGD(head.parameters(),
                                lr=params.lr,
                                momentum=params.momentum,
                                weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=1e-2, T_max=params.epochs
    )

    ckpt_dir = os.path.join(CKPT_ROOT, params.name)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(TB_ROOT, params.name))

    best_top1    = -1.0
    best_top5    = -1.0
    best_val_loss = float("inf")

    for epoch in range(params.epochs):
        print(f"\nEpoch {epoch+1}/{params.epochs}")

        train_m = train(train_loader, head, loss_fn, optimizer,
                        epoch, writer, params, device)
        scheduler.step()
        val_m   = test(test_loader, head, loss_fn,
                       epoch, writer, params, device)

        ckpt = {
            "epoch":     epoch,
            "model":     head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "params":    params.__dict__,
            "metrics": {**train_m, **val_m},
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pth"))

        if val_m["val_top1"] > best_top1:
            best_top1 = val_m["val_top1"]
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best_top1.pth"))

        if val_m["val_top5"] > best_top5:
            best_top5 = val_m["val_top5"]
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best_top5.pth"))

        if val_m["val_loss"] < best_val_loss:
            best_val_loss = val_m["val_loss"]
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_best_loss.pth"))

    print(f"\nDone. Best val top1: {100*best_top1:.2f}%  top5: {100*best_top5:.2f}%")
    writer.close()


if __name__ == "__main__":
    main()