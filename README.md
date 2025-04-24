# 🍽️ Calorie Estimation using Deep Learning
This project leverages a deep learning model—**MobileNetV2**—to estimate the calorie content of food items from images. It uses a food image dataset and performs extensive data preprocessing and augmentation to improve model generalization and accuracy.

## 🚀 Features

- Food image classification using **MobileNetV2**
- Calorie estimation based on predicted food category
- Data preprocessing and augmentation techniques applied for better model performance
- Web interface (Django/Flask) for uploading food images and viewing calorie predictions
- Achieved an accuracy of **85%**

## 🧠 Model Overview

We use **MobileNetV2**, a lightweight and efficient deep convolutional neural network architecture well-suited for image classification tasks. The model is trained to classify food items, and each label is mapped to a corresponding calorie estimate.

### 🔧 Preprocessing & Augmentation Techniques

To improve model accuracy and robustness, we apply multiple data augmentation techniques:
- **Random flipping** (horizontal/vertical)
- **Rotation**
- **Zoom**
- **Rescaling**
- **Shifting**

These help simulate real-world image variability and avoid overfitting.

## 📁 Project Structure

- `Food_Classification.ipynb` – Main notebook for training, testing, and evaluation
- `models.py` – Contains the MobileNetV2 model setup
- `views.py`, `urls.py` – Web interface routing and logic (likely Django or Flask)
- `Accuracy_curve_model.pdf`, `loss_curve_model.pdf`, `confusion_matrix_model.pdf` – Visual performance reports

## 📊 Performance Metrics

- **Accuracy**: ~85%
- Loss and accuracy trends during training and validation
- Confusion matrix visualization for class-wise evaluation

