"""
extract_features_dinov2.py

Run once per dataset split (train/test) to extract and cache DINOv2-large
CLS token features. The saved .pt files are the input to train_linear_head_dinov2.py.
"""

import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel
from CIFAR100_class import CIFAR100Dataset

BASE_DIR = "/data/user"
FEAT_DIR = os.path.join(BASE_DIR, "dinov2_features")
os.makedirs(FEAT_DIR, exist_ok=True)


DINOV2_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DEVICE = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def build_dataset(dataset_name: str, split: str):
    """
    Returns a Dataset that yields (image_tensor, y_fine, y_coarse).
    Swap this function out for NABirds / FGVC as needed.
    """
    assert split in ("train", "test")

    if dataset_name == "cifar100":
        raw = unpickle(f"cifar-100-python/{split}")
        # hce=False here — we just need raw images + labels, no hierarchy encoding
        dataset = CIFAR100Dataset(raw, transform=DINOV2_TRANSFORM, hce=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


@torch.no_grad()
def extract_features(dataset, save_path: str, batch_size: int = 512, num_workers: int = 8):
    backbone = AutoModel.from_pretrained("facebook/dinov2-large")
    backbone.eval().to(DEVICE)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_feats, all_fine_labels, all_coarse_labels = [], [], []

    for i, (x, y_fine, y_coarse) in enumerate(loader):
        out = backbone(pixel_values=x.to(DEVICE))
        cls = out.last_hidden_state[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        all_feats.append(cls.cpu())
        all_fine_labels.append(y_fine)
        all_coarse_labels.append(y_coarse)

        if i % 10 == 0:
            print(f"  batch {i+1}/{len(loader)}")

    feats        = torch.cat(all_feats)
    fine_labels  = torch.cat(all_fine_labels)
    coarse_labels = torch.cat(all_coarse_labels)

    torch.save(
        {"features": feats, "fine_labels": fine_labels, "coarse_labels": coarse_labels},
        save_path,
    )
    print(f"Saved {len(feats)} features → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="cifar100")
    parser.add_argument("--split",      default="train", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers",    type=int, default=8)
    args = parser.parse_args()

    save_path = os.path.join(FEAT_DIR, f"{args.dataset}_{args.split}_dinov2.pt")

    if os.path.exists(save_path):
        print(f"Features already exist at {save_path}, skipping extraction.")
        print("Delete the file to re-extract.")
        return

    dataset = build_dataset(args.dataset, args.split)
    extract_features(dataset, save_path,
                     batch_size=args.batch_size,
                     num_workers=args.workers)


if __name__ == "__main__":
    main()