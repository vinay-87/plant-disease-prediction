"""
Model architecture for Plant Disease Detection.
Uses EfficientNet-B0 with a custom classifier head.
"""

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, NUM_CLASSES


def create_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Create EfficientNet-B0 model with custom classifier.
    
    Uses timm library if available, otherwise falls back to torchvision.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pre-trained weights
    
    Returns:
        model: nn.Module
    """
    if HAS_TIMM:
        model = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
    else:
        # Fallback to torchvision
        from torchvision import models
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    
    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name and "head" not in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen backbone: {trainable:,} / {total:,} parameters trainable")


def unfreeze_backbone(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfrozen: all {total:,} parameters trainable")


def get_param_groups(model, lr: float, lr_backbone_factor: float = 0.1):
    """
    Create parameter groups with different learning rates.
    Backbone gets a lower LR than the classifier head.
    """
    classifier_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name or "head" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {"params": backbone_params, "lr": lr * lr_backbone_factor},
        {"params": classifier_params, "lr": lr},
    ]
