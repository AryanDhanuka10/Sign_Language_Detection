# 🤟 Sign Language Detection using AI

## 📌 Project Overview
Sign Language Detection using AI is a deep learning-based project designed to recognize and interpret sign language gestures. The model leverages Convolutional Neural Networks (CNNs) and other machine learning techniques to classify hand signs, enabling better communication for individuals with hearing impairments.

## 🚀 Features
- Converts sign language gestures into readable text
- Utilizes CNN for image-based classification
- Supports ONNX to Keras model conversion
- Deployable as a web application
- Multi-modal approach: text-based input and image-based predictions

## 🛠️ Technologies Used
- **Python** 🐍
- **TensorFlow & Keras** 🔥
- **OpenCV** 🎥
- **ONNX & ONNX2Keras** 🔄
- **Flask / Streamlit** 🌐 (for deployment)

## 📂 Project Structure
```
SignLanguageDetectionUsingAI/
│── dataset/             # Training and validation images
│── models/              # Trained model files
│── src/
│   ├── train.py         # Model training script
│   ├── predict.py       # Prediction script
│   ├── convert_to_h5.py # ONNX to Keras conversion script
│── web_app/
│   ├── app.py          # Web app for live detection
│── README.md
```

## 🎯 Installation & Setup
Follow these steps to set up the project:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AryanDhanuka10/SignLanguageDetectionUsingAI.git
cd SignLanguageDetectionUsingAI
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv sign_env
source sign_env/bin/activate  # On macOS/Linux
sign_env\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Convert ONNX to Keras (if needed)
```bash
python src/convert_to_h5.py
```

### 5️⃣ Run the Web App
```bash
python web_app/app.py
```

## 📸 Model Training
To train the model, run:
```bash
python src/train.py
```

## 🏆 Results & Performance
- CNN Validation Accuracy: **63.25%**
- Logistic Regression Accuracy: **72.14%**
- Improvements underway to enhance accuracy

## 🛠 Troubleshooting
- **ONNX to Keras conversion error?** Ensure `onnx2keras` is updated:
  ```bash
  pip install --upgrade onnx2keras
  ```
- **GPU not detected?** Disable secure boot or install proper NVIDIA drivers.

## 🤝 Contributing
Contributions are welcome! Feel free to fork, submit issues, or create pull requests.

## 📜 License
MIT License © 2025 Aryan Dhanuka

---
### ✨ Made with ❤️ by [AryanDhanuka10](https://github.com/AryanDhanuka10)

