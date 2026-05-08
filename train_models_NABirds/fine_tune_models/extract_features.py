import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel
from PIL import Image

BASE_DIR = "/data/user"
NAB_DIR  = os.path.join(BASE_DIR, "nabirds")
FEAT_DIR = os.path.join(BASE_DIR, "dinov2_features")
os.makedirs(FEAT_DIR, exist_ok=True)

DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class NABirdsDataset(Dataset):
    def __init__(self, nabirds_dir, split):
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
                path = os.path.join(nabirds_dir, "images", image_paths[img_id])
                self.samples.append((path, image_labels[img_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return TRANSFORM(Image.open(path).convert("RGB")), label


@torch.no_grad()
def extract(split):
    save_path = os.path.join(FEAT_DIR, f"nabirds_{split}_dinov2.pt")
    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
        return

    backbone = AutoModel.from_pretrained("facebook/dinov2-large")
    backbone.eval().to(DEVICE)

    loader = DataLoader(NABirdsDataset(NAB_DIR, split),
                        batch_size=512, shuffle=False,
                        num_workers=8, pin_memory=True)

    all_feats, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        out = backbone(pixel_values=x.to(DEVICE))
        cls = out.last_hidden_state[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        all_feats.append(cls.cpu())
        all_labels.append(y)
        if i % 10 == 0:
            print(f"  {split} batch {i+1}/{len(loader)}")

    torch.save({"features": torch.cat(all_feats),
                "fine_labels": torch.cat(all_labels)}, save_path)
    print(f"Saved -> {save_path}")


extract("train")
extract("test")