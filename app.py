#-----------------------------------
# STREAMLIT WEB APPLICATION SCRIPT
#-----------------------------------
import streamlit as st
import numpy as np
from PIL import Image
import pickle

# 1. Define mathematical activation functions required by the model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

# 2. Reconstruct the minimal NNClassifier class (Required for unpickling)
class NNClassifier(object):
    def getOutput(self, x):
        activation = x
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activation) + b
            if i == len(self.weights) - 1:
                activation = softmax(z) # Output layer
            else:
                activation = sigmoid(z) # Hidden layers
        return activation

    def predict(self, X_array):
        predictions = np.zeros(shape=len(X_array))
        for i, sample in enumerate(X_array):
            prediction = self.getOutput(sample.reshape(-1, 1))
            predictions[i] = np.argmax(prediction)
        return predictions

# 3. Load the exported model into the web app
@st.cache_resource
def load_model():
    with open('ovarian_mlp_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()
CATEGORIES = ['dominant_follicle', 'poly_cyst', 'complex_cyst', 'simple_cyst', 'healthy']

# 4. Build the Streamlit User Interface
st.title("Physiological & Pathological Ovarian Conditions Classifier 🩺🥼")
st.write("Upload an ultrasound image to receive an automated predictive diagnosis (physiological or pathological) based on our custom MLP architecture. There are 5 possibilities for prediction of ovarian conditions:. 1️⃣ Healthy Ovary, 2️⃣ Polycystic Ovary (PCOS), 3️⃣ Simple Cyst, 4️⃣ Complex Cyst, and 5️⃣Dominant Follicle.")

uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Ultrasound Image', use_container_width=True)
    
    if st.button('Predict Condition')
        with st.spinner('Analyzing image features...'):
            # Convert image to grayscale, resize, flatten, and normalize
            img_gray = image.convert('L')
            img_resized = img_gray.resize((64, 64))
            img_array = np.array(img_resized)
            flattened_array = img_array.flatten()
            normalized_array = flattened_array / 255.0
            input_data = np.array([normalized_array])
            
            # Execute prediction
            prediction_idx = int(model.predict(input_data)[0])
            pred_label = CATEGORIES[prediction_idx]
            
            st.success(f"Model Prediction: **{pred_label}**")