import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoModel
from PIL import Image

BASE_DIR  = "/data/user"
FGVC_DIR  = os.path.join(BASE_DIR, "fgvc-aircraft-2013b/fgvc-aircraft-2013b/data")
FEAT_DIR  = os.path.join(BASE_DIR, "dinov2_features")
os.makedirs(FEAT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FGVCDataset(Dataset):
    def __init__(self, fgvc_dir, split):
        variants = []
        with open(os.path.join(fgvc_dir, "variants.txt")) as f:
            for line in f:
                v = line.strip()
                if v:
                    variants.append(v)
        variant_to_idx = {v: i for i, v in enumerate(variants)}

        split_file_map = {
            "train":    "images_variant_train.txt",
            "val":      "images_variant_val.txt",
            "trainval": "images_variant_trainval.txt",
            "test":     "images_variant_test.txt",
        }

        self.samples = []
        with open(os.path.join(fgvc_dir, split_file_map[split])) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    img_id, variant_name = parts
                    label    = variant_to_idx[variant_name]
                    img_path = os.path.join(fgvc_dir, "images", f"{img_id}.jpg")
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return TRANSFORM(Image.open(path).convert("RGB")), label


@torch.no_grad()
def extract(split):
    save_path = os.path.join(FEAT_DIR, f"fgvc_{split}_dinov2.pt")
    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
        return

    backbone = AutoModel.from_pretrained("facebook/dinov2-large")
    backbone.eval().to(DEVICE)

    loader = DataLoader(FGVCDataset(FGVC_DIR, split),
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


extract("trainval")
extract("test")