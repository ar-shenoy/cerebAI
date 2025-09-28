# CerebAI: AI-Powered Stroke Detection System

##  Project Overview
CerebAI is a deep learning application designed to assist medical professionals by rapidly classifying CT scan images for the presence and type of stroke. Built on the advanced ConvNeXt architecture, the system provides a robust diagnosis coupled with a critical eXplainable AI (XAI) feature, ensuring predictions are transparent and medically intuitive.

**This project showcases high-performance multiclass classification and deployment readiness.**

##  Key Technical Achievements
| Metric | Score (Test Set) | Implication |
| :--- | :--- | :--- |
| **Mean IoU (mIoU)** | **~0.9843** | Top-tier performance for pixel-level prediction quality. |
| **Test F1 Score (Weighted)** | **~0.9805** | Excellent balance between Precision and Recall across all three classes. |
| **Model Architecture** | **ConvNeXt Base** | State-of-the-art model designed for robust feature extraction from medical images. |

##  Interpretability (XAI Feature)
The system uses **Integrated Gradients (IG)** from the Captum library to generate a heatmap overlay.

* **Function:** IG highlights the specific pixels that most strongly influence the model's final diagnosis.
* **Clinical Value:** This visual evidence helps doctors verify the prediction by confirming the model is focusing on the actual pathology (the stroke region) and not on noise or artifacts.

##  Deployment and Setup

### Local Run Instructions
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ar-shenoy/cerebAI.git
    cd cerebai_streamlit
    ```
2.  **Activate Environment:** Ensure your virtual environment is active.
    ```bash
    .\venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place Model Weights:** Ensure your trained model file (`best_model.pth`) is in the project root directory.
5.  **Launch App:**
    ```bash
    streamlit run cerebAI.py
    ```

### Streamlit Deployment (Cloud)
This repository is configured for one-click deployment on the Streamlit Community Cloud. The app is optimized to run on a shared CPU by limiting the Integrated Gradients calculation to **20 steps** to ensure fast performance.

---
