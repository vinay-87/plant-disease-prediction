"""
Grad-CAM visualization for Plant Disease Detection.
Generates heatmaps showing which regions the model focuses on.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE_STAGE2, BEST_MODEL_PATH

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


def get_inference_transform(img_size: int = IMG_SIZE_STAGE2):
    """Get the transform pipeline for inference."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalize a tensor image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def predict_with_gradcam(
    model, image_path: str, class_names: list, device=None
):
    """
    Predict disease and generate Grad-CAM heatmap.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to input image
        class_names: List of class names
        device: torch device
    
    Returns:
        dict with prediction, confidence, gradcam_overlay, original_image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass for prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()
    
    predicted_class = class_names[predicted_idx]
    
    # Top 5 predictions
    top5_probs, top5_indices = probabilities.topk(5)
    top5 = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top5_indices, top5_probs)
    ]
    
    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top5": top5,
        "original_image": image,
    }
    
    # Generate Grad-CAM
    if HAS_GRADCAM:
        try:
            # Get the target layer (last conv layer of EfficientNet)
            # For timm EfficientNet-B0
            if hasattr(model, "conv_head"):
                target_layer = model.conv_head
            elif hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                # Try to find last conv layer
                target_layer = None
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                
            if target_layer is not None:
                cam = GradCAM(
                    model=model,
                    target_layers=[target_layer],
                )
                
                targets = [ClassifierOutputTarget(predicted_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # Denormalize the image for overlay
                rgb_img = denormalize(input_tensor.squeeze(0).cpu())
                rgb_img = rgb_img.permute(1, 2, 0).numpy()
                
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                result["gradcam_overlay"] = cam_image
                result["gradcam_heatmap"] = grayscale_cam
            else:
                result["gradcam_overlay"] = None
                result["gradcam_heatmap"] = None
                
        except Exception as e:
            print(f"  Grad-CAM failed: {e}")
            result["gradcam_overlay"] = None
            result["gradcam_heatmap"] = None
    else:
        result["gradcam_overlay"] = None
        result["gradcam_heatmap"] = None
    
    return result


def visualize_prediction(result: dict, save_path: str = None):
    """Visualize prediction with Grad-CAM overlay."""
    has_gradcam = result.get("gradcam_overlay") is not None
    ncols = 3 if has_gradcam else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    
    if ncols == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(result["original_image"])
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    
    if has_gradcam:
        # Grad-CAM overlay
        axes[1].imshow(result["gradcam_overlay"])
        axes[1].set_title("Grad-CAM Overlay", fontsize=14)
        axes[1].axis("off")
        
        # Heatmap only
        axes[2].imshow(result["gradcam_heatmap"], cmap="jet")
        axes[2].set_title("Attention Heatmap", fontsize=14)
        axes[2].axis("off")
    
    # Add prediction text
    pred_text = (
        f"Prediction: {result['predicted_class']}\n"
        f"Confidence: {result['confidence']:.1%}"
    )
    fig.suptitle(pred_text, fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Visualization saved to {save_path}")
    
    plt.close()
    return fig
