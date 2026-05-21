# Ovarian Ultrasound Pathology Classification: Custom MLP vs. Vision Transformer (ViT)

This repository contains the complete source code, evaluation metrics, and an optional production deployment interface for classifying ovarian pathologies from ultrasound images. This project implements a comparative analysis between a foundational matrix-driven network and a state-of-the-art transformer architecture.

---

## 📂 Repository Components

### 1. Main Source Code
* **`AI_Final_Project_MLPvsTransformers_Cazañas.ipynb`**: The complete core development pipeline. It contains:
  * **Part I - Custom MLP Base:** Full implementation from scratch using NumPy (forward feed matrices, backpropagation derivative calculus loops, and Sigmoid/Softmax layer tracking). Achieves a **97.00% global test accuracy**.
  * **Part II - Vision Transformer Contrast:** Transfer learning optimization utilizing a `vit_small_patch16_224` backbone architecture fine-tuned via PyTorch and `timm`. Achieves a **100.00% global test accuracy**.

### 2. Optional Production Interface (Extra Component)
* **`app.py`**: A lightweight user interface built with Streamlit designed for live clinical image testing.
  * ⚠️ *Note: This interactive application is an optional project extension and operates **exclusively** on top of the custom NumPy MLP weights parameters.*
* **`ovarian_mlp_model.pkl`**: The serialized binary file containing the final trained mathematical weights and biases arrays of the MLP model.
* **`requirements.txt`**: The virtual web server configuration file listing the libraries required to execute the Streamlit application interface.

---

## 📊 Dataset Specifications & Download Instructions

* **Dataset Name:** Ovarian Ultrasound Image Dataset
* **Source Download Link:** Available via Kaggle at [Kaggle Ovarian Dataset URL](https://www.kaggle.com/datasets/ucimachinelearning/ovarian-ultrasound-image-dataset)
* **Dataset Scale:** 6,876 high-resolution ultrasound images divided across 5 diagnostic tissue categories:
  1. `dominant_follicle`
  2. `poly_cyst`
  3. `complex_cyst`
  4. `simple_cyst`
  5. `healthy`

### Dataset Loading Setup
To run the source notebook, ensure you place your personal Kaggle API credential token file (`kaggle.json`) into your execution environment directory to trigger automated dataset retrieval and extraction blocks.

---

## 💻 Execution Steps

### Option A: Executing the Core Development Notebook (`.ipynb`)
1. Open Google Colab or a local Jupyter Notebook instance.
2. Upload the `AI_Final_Project_MLPvsTransformers_Cazañas.ipynb` file.
3. Upload your Kaggle credential token file (`kaggle.json`) when prompted in the initial storage environment cells.
4. Run all cells sequentially. The code will automatically download the dataset, execute the stratified 70/15/15 split, initialize training configurations, and generate evaluation plots.

### Option B: Running the Optional Streamlit App Locally
To test the optional interactive web interface locally on your system, execute these commands inside your terminal:

```bash
# 1. Clone this repository to your machine
git clone [https://github.com/emiliaveronica14-ia/ovarian-ultrasound-app-mlp.git](https://github.com/emiliaveronica14-ia/ovarian-ultrasound-app-mlp.git)

# 2. Navigate into the project folder
cd ovarian-ultrasound-mlp-app

# 3. Install the required runtime libraries
pip install -r requirements.txt

# 4. Boot up the local web deployment server
streamlit run app.py
