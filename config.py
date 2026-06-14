"""
Configuration for Plant Disease Detection project.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "plantvillage")
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

# Training hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS_FROZEN = 5        # Train only classifier head
EPOCHS_FINETUNE = 25     # Fine-tune entire model
TOTAL_EPOCHS = EPOCHS_FROZEN + EPOCHS_FINETUNE

# Optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

# Image sizes (progressive resizing)
IMG_SIZE_STAGE1 = 128
IMG_SIZE_STAGE2 = 224

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 38  # PlantVillage has 38 classes

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
