import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional

# --- CONFIGURATION ---
MODEL_PATH = "best_model.pth"
CLASS_LABELS = ['No Stroke', 'Ischemic Stroke', 'Hemorrhagic Stroke']
IMAGE_SIZE = 224
# Use CPU by default for stability in free deployment, but change this locally to 'cuda' for speed!
DEVICE = torch.device("cpu") 

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads the model architecture and saved weights."""
    try:
        model = timm.create_model('convnext_base', pretrained=False)
        model.reset_classifier(num_classes=len(CLASS_LABELS))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model. Check model file and path. Error: {e}")
        return None

# --- HELPER FUNCTIONS ---

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalizes a PyTorch tensor for matplotlib visualization."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)

def preprocess_image(image_bytes: bytes) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Loads, resizes, and normalizes the image for model input."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None: return None, None
    image_rgb = cv2.cvtColor(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_GRAY2RGB)
    
    image_norm = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    input_tensor = torch.tensor(image_norm, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    
    return input_tensor.to(DEVICE), image_rgb

def generate_attribution(model: nn.Module, input_tensor: torch.Tensor, predicted_class_idx: int, n_steps: int = 20) -> np.ndarray:
    """Computes Integrated Gradients for the given input and class."""
    
    # CRITICAL FIX: Captum requires standard Python int, not numpy.int64
    target_class_int = int(predicted_class_idx) 
    
    # CRITICAL: Enables gradient tracking for Captum
    input_tensor.requires_grad_(True) 
    
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(DEVICE)

    attributions_ig = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target_class_int,
        n_steps=n_steps # Using dynamic or default steps
    )
    
    # Process Attributions: Sum across color channels and normalize the heatmap
    attributions_ig_vis = attributions_ig.squeeze(0).sum(dim=0).abs().cpu().detach().numpy()
    
    if attributions_ig_vis.max() > 0:
        attributions_ig_vis = attributions_ig_vis / attributions_ig_vis.max()
    
    return attributions_ig_vis

def plot_heatmap_and_original(original_image: np.ndarray, heatmap: np.ndarray, predicted_label: str):
    """Creates a Matplotlib figure for visualization."""
    
    # Use dynamic sizing for better responsiveness
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) 
    
    # Convert image to 0-1 range for plotting
    original_image_vis = (original_image.astype(np.float32) / 255.0) 

    # --- Plot 1: Original Image ---
    ax1.imshow(original_image_vis)
    ax1.set_title("Original CT Scan", fontsize=14)
    ax1.axis('off')

    # --- Plot 2: Integrated Gradients ---
    ax2.imshow(original_image_vis)
    
    # Dynamic alpha mask: fades out non-contributing regions
    alpha_mask = heatmap * 0.7 + 0.3 

    # Aesthetic Fix: Use 'jet' colormap for clinical highlight (red/yellow)
    ax2.imshow(heatmap, cmap='jet', alpha=alpha_mask, vmin=0, vmax=1)
    ax2.set_title(f"Interpretation: {predicted_label}", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# -------------------- STREAMLIT FRONTEND --------------------


st.set_page_config(page_title="CerebAI: Stroke Prediction Dashboard", layout="wide")
st.title("CerebAI: AI-Powered Stroke Detection")
st.markdown("---")

# Load the model
model = load_model(MODEL_PATH)

if model is not None:
    # --- INTERACTIVE CONTROLS (Sidebar or Main Area) ---
    st.markdown("### Analysis Controls")
    
    n_steps_slider = st.slider(
        'Integration Steps (Affects Accuracy & Speed)',
        min_value=5, 
        max_value=50, 
        value=20, # Default to a safe, medium-speed value
        step=5,
        help="Higher steps (up to 50) provide a smoother, more accurate heatmap but use more CPU."
    )
    st.markdown("---")


    # --- FILE UPLOAD ---
    st.markdown("### Upload CT Scan Image")
    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or JPEG file", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        
        # --- DISPLAY AND RESULTS LAYOUT ---
        col1, col2 = st.columns(2) # Retaining old columns structure for familiar look

        with col1:
            st.subheader("Uploaded Image")
            st.image(image_bytes, use_container_width=True) # Responsive fix

        # Run Prediction and Attribution
        input_tensor, original_image_rgb = preprocess_image(image_bytes)
        
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
            
            with col2:
                st.subheader("Prediction Summary")
                
                # Metric based on prediction
                st.metric(
                    label="Diagnosis", 
                    value=predicted_label,
                    delta=f"{confidence_score*100:.2f}% Confidence",
                    delta_color='normal' # Let Streamlit choose color
                )
                
                st.markdown("---")
                st.subheader("Confidence Breakdown")
                
                # Display probabilities in a clean, professional table
                prob_data = {
                    'Class': CLASS_LABELS,
                    'Confidence': [f"{p:.4f}" for p in probabilities]
                }
                st.dataframe(prob_data, hide_index=True, use_container_width=True)

            # --- PLOT INTERPRETATION ---
            st.markdown("---")
            st.subheader("Model Interpretation (Integrated Gradients)")
            
            fig = plot_heatmap_and_original(original_image_rgb, heatmap, predicted_label)
            st.pyplot(fig, clear_figure=True, use_container_width=True) # Responsive Plot

            st.success("Analysis Complete: The heatmap highlights the regions most critical to the diagnosis.")