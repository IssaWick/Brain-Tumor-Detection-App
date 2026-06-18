# app/app.py — Brain Tumor Detection Dashboard

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# ---- Page config ----
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

IMG_SIZE = (224, 224)

# ---- Load model and labels (cached so it only loads once) ----
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "..", "models", "brain_tumor_vgg16.keras")
    labels_path = os.path.join(base, "..", "models", "class_names.json")
    model = tf.keras.models.load_model(model_path)
    with open(labels_path) as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()

# ---- Preprocess an uploaded image exactly like training ----
def preprocess(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)   # ensure 3 channels + correct size
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)                       # SAME preprocessing as training
    arr = np.expand_dims(arr, axis=0)                 # add batch dimension -> (1,224,224,3)
    return arr

# ---- UI ----
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image and the model will classify it into one of four categories.")

# Human-readable label names
display_names = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor"
}

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            x = preprocess(image)
            probs = model.predict(x)[0]            # shape (4,)
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]
            confidence = float(probs[pred_idx]) * 100

        # Main result
        st.success(f"**Prediction: {display_names[pred_label]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Full probability breakdown
        st.subheader("All class probabilities")
        prob_dict = {
            display_names[class_names[i]]: float(probs[i])
            for i in range(len(class_names))
        }
        st.bar_chart(prob_dict)

        # Caution note for the hard classes
        if pred_label in ("glioma", "meningioma") and confidence < 70:
            st.warning(
                "Glioma and meningioma can look similar on MRI. "
                "This prediction has lower confidence — interpret with care."
            )

st.caption("Educational/portfolio project. Not for clinical or diagnostic use.")