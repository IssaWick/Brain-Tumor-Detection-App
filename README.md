# 🧠 Brain Tumor Detection

A deep learning web app that classifies brain MRI images into four categories:
**glioma, meningioma, no tumor, and pituitary tumor**.

## Model
- Transfer learning with **VGG16** (ImageNet weights, frozen base)
- Custom classification head (Dense + Dropout + Softmax)
- Trained on a balanced dataset (1,400 images/class)
- **~87% test accuracy** on 1,600 unseen images

## Run locally
```bash
conda create -n braintumor python=3.10 -y
conda activate braintumor
pip install -r requirements.txt
streamlit run app/app.py
```

## Project structure
- `notebooks/` — training & evaluation notebook
- `app/` — Streamlit dashboard
- `models/` — trained model + class labels

> ⚠️ Educational/portfolio project. Not for clinical or diagnostic use.