# 🧠 Brain Tumor Classification Web App (VGG16 + Grad-CAM)

A full-stack deep learning web application for brain tumor classification using MRI images. This project uses a pretrained VGG16 model for prediction and Grad-CAM for visual explanations.

> Upload a brain MRI → Predict tumor type → Visualize activated region using Grad-CAM heatmap.

---

## 🔍 Features

- ✅ Real-time image upload via web interface
- ✅ VGG16 model for high-accuracy predictions
- ✅ Grad-CAM heatmap for explainability
- ✅ Clean, modern frontend (Google IDX inspired)
- ✅ Built with Flask, TensorFlow, HTML/CSS
- ✅ Deployable on Colab or locally

---

## 🧠 Tumor Classes

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

---

## ⚠️ Model File Not Included (100MB GitHub Limit)

Due to GitHub’s 100MB file upload limit, the trained model file (`vgg16_model.h5`) is not stored in this repository.

### 📦 Download the Model

👉 [Download `vgg16_model.h5`](https://drive.google.com/your-download-link)

---

## 🗂 How to Use the Model

### 🔁 Option A: Google Colab (Recommended)

1. Upload `vgg16_model.h5` to your Colab session manually:
```python
from google.colab import files
uploaded = files.upload()  # Upload the .h5 file

Confirm it’s saved:
import os
assert os.path.exists("vgg16_model.h5"), "Model not found"

 Running the App
1.Install dependencies:
pip install flask pyngrok tensorflow numpy opencv-python matplotlib

2.	Set your Ngrok Authtoken:
from pyngrok import ngrok
ngrok.set_auth_token("your_token_here")

3.	Run the Flask app:
python app.py

├── app.py
├── vgg16_model.h5      <- download & place manually
├── gradcam_utils.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── gradcam.jpg
├── uploads/
└── README.md

	•	Built with ❤️ by Chaitanya
	•	MRI dataset used: Brain Tumor Classification Dataset (Kaggle)
