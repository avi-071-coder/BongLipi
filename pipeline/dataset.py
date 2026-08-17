"""
=============================================================================
BONGLIPI - Step 3: Dataset Loader & Data Augmentation
=============================================================================
PyTorch dataset class for loading the segmented Bengali characters.
Includes augmentation for training robustness.
=============================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image


class BengaliCharDataset(Dataset):
    """
    Dataset for Bengali handwritten character images.
    Each image is a 64x64 grayscale PNG stored in class_XXX/ directories.
    """
    
    def __init__(self, root_dir, transform=None, target_size=64):
        self.root = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Load label map
        with open(self.root / "label_map.json", "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        
        self.num_classes = int(self.label_map["num_classes"])
        self.id_to_char = self.label_map["id_to_char"]
        self.class_list = self.label_map["class_list"]
        
        # Collect all samples
        self.samples = []  # [(filepath, class_id), ...]
        
        for cls_id in range(self.num_classes):
            cls_dir = self.root / f"class_{cls_id:03d}"
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), cls_id))
        
        print(f"BengaliCharDataset: {len(self.samples)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: return zeros
            img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        
        # Resize if needed
        if img.shape != (self.target_size, self.target_size):
            img = cv2.resize(img, (self.target_size, self.target_size))
        
        # Convert to PIL for transforms
        img = Image.fromarray(img, mode='L')
        
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        
        return img, label
    
    def get_char_for_id(self, class_id):
        return self.id_to_char.get(str(class_id), "?")


def get_transforms(train=True, img_size=64):
    """Get data augmentation transforms."""
    if train:
        return T.Compose([
            T.RandomAffine(
                degrees=10,           # Small rotation
                translate=(0.08, 0.08),  # Small shift
                scale=(0.9, 1.1),     # Small scale
                shear=5,              # Small shear
            ),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])


def create_dataloaders(data_dir, batch_size=64, val_split=0.15, test_split=0.1, 
                        num_workers=0, seed=42):
    """
    Create train/val/test dataloaders with proper splits.
    """
    # Full dataset (no augmentation for splitting)
    full_dataset = BengaliCharDataset(data_dir)
    
    n = len(full_dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    
    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    # Apply transforms
    train_dataset = TransformSubset(train_set, get_transforms(train=True))
    val_dataset = TransformSubset(val_set, get_transforms(train=False))
    test_dataset = TransformSubset(test_set, get_transforms(train=False))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    
    return train_loader, val_loader, test_loader, full_dataset.num_classes, full_dataset.label_map


class TransformSubset(Dataset):
    """Wrapper to apply transforms to a Subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is already a PIL Image from BengaliCharDataset.__getitem__
        # But since subset returns already-transformed data, we need to handle this
        # The base dataset returns a tensor, so we need a different approach
        
        # Get the original sample from the underlying dataset
        original_idx = self.subset.indices[idx]
        img_path, label = self.subset.dataset.samples[original_idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))
        
        img = Image.fromarray(img, mode='L')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


if __name__ == "__main__":
    # Test the dataloader
    train_loader, val_loader, test_loader, n_classes, lm = create_dataloaders("dataset")
    
    # Check a batch
    for batch_imgs, batch_labels in train_loader:
        print(f"Batch shape: {batch_imgs.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Label range: {batch_labels.min()} - {batch_labels.max()}")
        print(f"Pixel range: {batch_imgs.min():.2f} - {batch_imgs.max():.2f}")
        break

# Day 2: Dataset docs
