"""
Streamlit web application for Plant Disease Detection.
Upload a leaf image and get instant disease diagnosis with Grad-CAM explanations.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import joblib
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BEST_MODEL_PATH, MODEL_DIR, IMG_SIZE_STAGE2
from src.model import create_model
from src.gradcam import predict_with_gradcam, get_inference_transform


# Disease information database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "description": "Fungal disease caused by Venturia inaequalis.",
        "treatment": "Apply fungicides, remove fallen leaves, prune for air circulation."
    },
    "Apple___Black_rot": {
        "disease": "Black Rot",
        "description": "Caused by Botryosphaeria obtusa fungus.",
        "treatment": "Remove mummified fruits, prune dead wood, apply fungicides."
    },
    "Apple___healthy": {
        "disease": "Healthy",
        "description": "No disease detected.",
        "treatment": "Continue regular care and monitoring."
    },
    "Tomato___Late_blight": {
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans.",
        "treatment": "Apply copper-based fungicides, remove infected plants."
    },
    "Tomato___healthy": {
        "disease": "Healthy",
        "description": "No disease detected.",
        "treatment": "Continue regular care."
    },
}


@st.cache_resource
def load_model_cached():
    """Load model and class names."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names_path = os.path.join(MODEL_DIR, "class_names.joblib")
    
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(class_names_path):
        class_names = joblib.load(class_names_path)
        model = create_model(num_classes=len(class_names), pretrained=False)
        
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        return model, class_names, device
    
    return None, None, device


def format_class_name(class_name: str) -> str:
    """Format class name for display."""
    parts = class_name.split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ")
        return f"{plant} — {condition}"
    return class_name.replace("_", " ")


def main():
    st.set_page_config(
        page_title="Plant Disease Detector",
        page_icon="🌿",
        layout="wide",
    )
    
    st.title("🌿 Plant Disease Detection")
    st.markdown(
        "Upload a photo of a plant leaf to identify diseases using deep learning. "
        "The model analyzes the image and provides a diagnosis with visual explanations."
    )
    
    model, class_names, device = load_model_cached()
    
    if model is None:
        st.error(
            "⚠️ Model not found! Please train the model first:\n"
            "```bash\npython -m src.train\n```"
        )
        st.info(
            "Make sure to download the PlantVillage dataset from "
            "[Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) "
            "and place it in `data/plantvillage/`"
        )
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📸 Upload a leaf image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear photo of a single plant leaf"
    )
    
    if uploaded_file is not None:
        # Save temporarily
        temp_path = os.path.join(MODEL_DIR, "temp_upload.jpg")
        image = Image.open(uploaded_file).convert("RGB")
        image.save(temp_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_column_width=True)
        
        # Predict
        with st.spinner("🔍 Analyzing..."):
            result = predict_with_gradcam(model, temp_path, class_names, device)
        
        with col2:
            st.subheader("🎯 Diagnosis")
            
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            
            # Color code confidence
            if confidence > 0.9:
                color = "green"
            elif confidence > 0.7:
                color = "orange"
            else:
                color = "red"
            
            formatted_name = format_class_name(predicted_class)
            st.markdown(f"### {formatted_name}")
            st.markdown(f"**Confidence:** :{color}[{confidence:.1%}]")
            
            # Disease info
            if predicted_class in DISEASE_INFO:
                info = DISEASE_INFO[predicted_class]
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
            
            # Grad-CAM
            if result.get("gradcam_overlay") is not None:
                st.subheader("🔬 Model Attention (Grad-CAM)")
                st.image(
                    result["gradcam_overlay"],
                    caption="Red regions show where the model is focusing",
                    use_column_width=True
                )
        
        # Top 5 predictions
        st.subheader("📊 Top 5 Predictions")
        for i, (cls_name, prob) in enumerate(result["top5"], 1):
            formatted = format_class_name(cls_name)
            st.progress(prob, text=f"{i}. {formatted} — {prob:.1%}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    else:
        # Show sample usage
        st.info(
            "👆 Upload a leaf image above to get started. "
            "The model supports 38 different plant disease classes across "
            "multiple crops including Apple, Tomato, Grape, Corn, and more."
        )
        
        st.subheader("📋 Supported Plants & Diseases")
        if class_names:
            plants = set()
            for cn in class_names:
                parts = cn.split("___")
                if len(parts) == 2:
                    plants.add(parts[0].replace("_", " "))
            
            cols = st.columns(4)
            for i, plant in enumerate(sorted(plants)):
                cols[i % 4].markdown(f"- 🌱 {plant}")
    
    # Sidebar
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown(
        "This application uses **EfficientNet-B0** "
        "fine-tuned on the PlantVillage dataset with "
        "**96.4% accuracy** across 38 disease classes."
    )
    st.sidebar.markdown("### 🛠 Tech Stack")
    st.sidebar.markdown("- PyTorch + EfficientNet-B0")
    st.sidebar.markdown("- Grad-CAM for interpretability")
    st.sidebar.markdown("- Streamlit for deployment")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built by [Vinay Kumar](https://github.com/vinay-87)"
    )


if __name__ == "__main__":
    main()
