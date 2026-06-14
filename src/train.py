"""
Training loop for Plant Disease Detection.
Implements two-phase training: frozen backbone + fine-tuning.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LEARNING_RATE, WEIGHT_DECAY, MIN_LR, BEST_MODEL_PATH,
    EPOCHS_FROZEN, EPOCHS_FINETUNE, TOTAL_EPOCHS, BATCH_SIZE,
    IMG_SIZE_STAGE1, IMG_SIZE_STAGE2, MODEL_DIR
)
from src.dataset import create_data_loaders
from src.model import create_model, freeze_backbone, unfreeze_backbone, get_param_groups


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train():
    """Main training pipeline with two-phase training."""
    print("=" * 60)
    print("PLANT DISEASE DETECTION — TRAINING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Phase 1: Train with frozen backbone at lower resolution
    print(f"\n{'='*60}")
    print(f"PHASE 1: Frozen Backbone (epochs 1-{EPOCHS_FROZEN}, size={IMG_SIZE_STAGE1})")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        img_size=IMG_SIZE_STAGE1, batch_size=BATCH_SIZE
    )
    
    model = create_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)
    freeze_backbone(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FROZEN, eta_min=MIN_LR)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, EPOCHS_FROZEN + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(
            f"  Epoch {epoch}/{EPOCHS_FROZEN} — "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, BEST_MODEL_PATH)
    
    # Phase 2: Fine-tune entire model at full resolution
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning (epochs {EPOCHS_FROZEN+1}-{TOTAL_EPOCHS}, size={IMG_SIZE_STAGE2})")
    print("=" * 60)
    
    # Reload data at higher resolution
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        img_size=IMG_SIZE_STAGE2, batch_size=BATCH_SIZE
    )
    
    unfreeze_backbone(model)
    
    # Differential learning rates
    param_groups = get_param_groups(model, lr=LEARNING_RATE * 0.1, lr_backbone_factor=0.1)
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE, eta_min=MIN_LR)
    
    for epoch in range(EPOCHS_FROZEN + 1, TOTAL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(
            f"  Epoch {epoch}/{TOTAL_EPOCHS} — "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, BEST_MODEL_PATH)
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f})")
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} (epoch {checkpoint['epoch']})")
    
    # Save history
    import json
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Save class names
    import joblib
    joblib.dump(class_names, os.path.join(MODEL_DIR, "class_names.joblib"))
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {BEST_MODEL_PATH}")
    print(f"   History: {history_path}")
    
    return model, history


if __name__ == "__main__":
    train()
