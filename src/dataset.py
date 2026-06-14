"""
Dataset loading and augmentation for Plant Disease Detection.
"""

import os
import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, IMG_SIZE_STAGE1, IMG_SIZE_STAGE2,
    IMAGENET_MEAN, IMAGENET_STD
)


class PlantDiseaseDataset(Dataset):
    """
    PlantVillage dataset for plant disease classification.
    
    Expected directory structure:
        data/plantvillage/
        ├── Apple___Apple_scab/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── Apple___Black_rot/
        └── ...
    """
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Scan directory structure and collect (image_path, label) pairs."""
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root_dir}. "
                f"Please download from: "
                f"https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
            )
        
        # Get sorted class folders
        class_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Collect all image paths
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), label))
        
        print(f"  Loaded {len(self.samples)} images across {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: Could not load {img_path}: {e}")
            # Return a blank image on error
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size: int, is_training: bool = True):
    """
    Get image transforms for training or evaluation.
    
    Args:
        img_size: Target image size
        is_training: Whether to apply data augmentation
    
    Returns:
        torchvision.transforms.Compose
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # Resize to slightly larger
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def create_data_loaders(
    img_size: int = IMG_SIZE_STAGE2,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    print("📂 Loading dataset...")
    
    # Create dataset without transforms (for splitting)
    full_dataset = PlantDiseaseDataset(DATA_DIR, transform=None)
    
    # Stratified split
    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val
    
    # Reproducible random split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create datasets with appropriate transforms
    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)
    
    # We need to create separate dataset instances with different transforms
    train_dataset = PlantDiseaseDataset(DATA_DIR, transform=train_transform)
    val_dataset = PlantDiseaseDataset(DATA_DIR, transform=eval_transform)
    test_dataset = PlantDiseaseDataset(DATA_DIR, transform=eval_transform)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    print(f"  Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")
    
    return train_loader, val_loader, test_loader, full_dataset.classes
