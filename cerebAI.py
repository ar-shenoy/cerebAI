import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os 
import requests 
import pydicom # REQUIRED FOR DICOM SUPPORT
import io 
import gc # For memory management

# --- CONFIGURATION ---
# >>> IMPORTANT: REPLACE THIS URL WITH YOUR ACTUAL HUGGING FACE LINK! <<<
HF_MODEL_URL = "https://huggingface.co/arshenoy/cerebAI-stroke-model/resolve/main/best_model.pth" 
DOWNLOAD_MODEL_PATH = "best_model_cache.pth"
CLASS_LABELS = ['No Stroke', 'Ischemic Stroke', 'Hemorrhagic Stroke']
IMAGE_SIZE = 224
DEVICE = torch.device("cpu") # For Streamlit Cloud stability

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_url, local_path):
    """Downloads model from URL if not cached, and loads the weights."""
    
    if not os.path.exists(local_path):
        st.info(f"Model not found locally. Downloading from remote repository...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status() 
            
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model download complete!")
        except Exception as e:
            st.error(f"FATAL ERROR: Could not download model. Check the URL. Error: {e}")
            return None

    try:
        model = timm.create_model('convnext_base', pretrained=False)
        model.reset_classifier(num_classes=len(CLASS_LABELS))
        model.load_state_dict(torch.load(local_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model weights from cache. Error: {e}")
        return None

# --- HELPER FUNCTIONS ---

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalizes a PyTorch tensor for matplotlib visualization."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0).detach() 
    else:
        tensor = tensor.detach() 
    
    mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)

def preprocess_image(image_bytes: bytes, file_name: str) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Loads, processes, and normalizes image, handling DICOM or JPG/PNG."""
    
    # 1. READ IMAGE DATA (Handles DICOM vs Standard formats)
    if file_name.lower().endswith(('.dcm', '.dicom')):
        try:
            dcm = pydicom.dcmread(io.BytesIO(image_bytes))
            
            # FIX: Convert to Hounsfield Units (HU)
            pixel_array = dcm.pixel_array.astype(np.int16)
            slope = dcm.RescaleSlope
            intercept = dcm.RescaleIntercept
            pixel_array = pixel_array * slope + intercept
            
            # Apply Standard Brain Window (-100 HU to 150 HU)
            window_center = 40 
            window_width = 150
            min_hu = window_center - (window_width / 2)
            max_hu = window_center + (window_width / 2)
            
            # Apply the windowing transformation and scale to 0-255
            pixel_array[pixel_array < min_hu] = min_hu
            pixel_array[pixel_array > max_hu] = max_hu
            image_grayscale = ((pixel_array - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
            
        except Exception:
            return None, None
    else:
        # Read standard image (PNG/JPG)
        image_grayscale = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image_grayscale is None: return None, None
            
    # 2. STANDARD PREPROCESSING
    image_rgb = cv2.cvtColor(cv2.resize(image_grayscale, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2RGB)
    
    image_norm = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    input_tensor = torch.tensor(image_norm, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    
    return input_tensor.to(DEVICE), image_rgb

def generate_attribution(model: nn.Module, input_tensor: torch.Tensor, predicted_class_idx: int, n_steps: int = 20) -> np.ndarray:
    """Computes Integrated Gradients for the given input and class."""
    target_class_int = int(predicted_class_idx) 
    input_tensor.requires_grad_(True) 
    
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(DEVICE)

    attributions_ig = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target_class_int,
        n_steps=n_steps
    )
    
    attributions_ig_vis = attributions_ig.squeeze(0).sum(dim=0).abs().cpu().detach().numpy()
    
    if attributions_ig_vis.max() > 0:
        attributions_ig_vis = attributions_ig_vis / attributions_ig_vis.max()
    
    return attributions_ig_vis

def plot_heatmap_and_original(original_image: np.ndarray, heatmap: np.ndarray, predicted_label: str):
    """Creates a Matplotlib figure for visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) 
    original_image_vis = (original_image.astype(np.float32) / 255.0) 

    ax1.imshow(original_image_vis)
    ax1.set_title("Original CT Scan", fontsize=14)
    ax1.axis('off')

    ax2.imshow(original_image_vis)
    alpha_mask = heatmap * 0.7 + 0.3 

    ax2.imshow(heatmap, cmap='jet', alpha=alpha_mask, vmin=0, vmax=1)
    ax2.set_title(f"Interpretation: {predicted_label}", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# ==============================================================================
# -------------------- STREAMLIT FRONTEND --------------------
# ==============================================================================

st.set_page_config(page_title="CerebAI: Stroke Prediction Dashboard", layout="wide")
st.title("CerebAI: AI-Powered Stroke Detection")
st.markdown("---")

# FIX: Load the model using the download mechanism
model = load_model(HF_MODEL_URL, DOWNLOAD_MODEL_PATH)

if model is not None:
    # --- INTERACTIVE CONTROLS (Sidebar or Main Area) ---
    st.markdown("### Analysis Controls")
    
    n_steps_slider = st.slider(
        'Integration Steps (Affects Accuracy & Speed)',
        min_value=5, 
        max_value=50, 
        value=20, 
        step=5,
        help="Higher steps (up to 50) provide a smoother, more accurate heatmap but use more CPU."
    )
    st.markdown("---")


    # --- FILE UPLOAD ---
    st.markdown("### Upload CT Scan Image")
    uploaded_file = st.file_uploader(
        "Choose a Dicom, PNG, JPG, or JPEG file", 
        type=["dcm", "dicom", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        file_name = uploaded_file.name 
        
        # 1. PROCESS IMAGE FIRST (Defines original_image_rgb)
        input_tensor, original_image_rgb = preprocess_image(image_bytes, file_name) 

        # --- DISPLAY AND RESULTS LAYOUT ---
        col1, col2 = st.columns(2) 

        with col1:
            st.subheader("Uploaded Image")
            # Display the processed NumPy array
            st.image(original_image_rgb, use_container_width=True, caption=file_name) 

        # Run Prediction and Attribution
        if input_tensor is not None:
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
                predicted_class_idx = np.argmax(probabilities)
            
            predicted_label = CLASS_LABELS[predicted_class_idx]
            confidence_score = probabilities[predicted_class_idx]
            
            # Generate Attribution
            heatmap = generate_attribution(model, input_tensor, predicted_class_idx, n_steps=n_steps_slider)
            
            # CRITICAL MEMORY MANAGEMENT
            del input_tensor 
            del output
            gc.collect() 

            with col2:
                st.subheader("Prediction Summary")
                
                st.metric(
                    label="Diagnosis", 
                    value=predicted_label,
                    delta=f"{confidence_score*100:.2f}% Confidence",
                    delta_color='normal' 
                )
                
                st.markdown("---")
                st.subheader("Confidence Breakdown")
                
                prob_data = {
                    'Class': CLASS_LABELS,
                    'Confidence': [f"{p:.4f}" for p in probabilities]
                }
                st.dataframe(prob_data, hide_index=True, use_container_width=True)

            # --- PLOT INTERPRETATION ---
            st.markdown("---")
            st.subheader("Model Interpretation (Integrated Gradients)")
            
            fig = plot_heatmap_and_original(original_image_rgb, heatmap, predicted_label)
            st.pyplot(fig, clear_figure=True, use_container_width=True) 

            st.success("Analysis Complete: The heatmap highlights the regions most critical to the diagnosis.")
