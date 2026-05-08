import numpy as np
from torch.utils.data import Dataset

class CIFAR100Dataset(Dataset):
    def __init__(self, data_dict, transform=None, hce=False):
        self.images = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.fine_labels = np.array(data_dict[b'fine_labels'])
        self.coarse_labels = np.array(data_dict[b'coarse_labels'])
        self.transform = transform
        self.hce = hce
        
    def __len__(self):
        return len(self.fine_labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        fine_label = self.fine_labels[idx]
        coarse_label = self.coarse_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Offset fine labels by 20 to match your indexing scheme
        if self.hce:
            fine_label = fine_label + 20
            
        return image, fine_label, coarse_label