# 🖐️ Sign Language Detection

A deep learning-based **Sign Language Detection** project that classifies American Sign Language (ASL) hand gestures.

## 🚀 Live Demo
🔗 **[Sign Language Detection App](https://signlanguagedetection-gcwbgbyxnlkgorwbdhaukj.streamlit.app/)**

---

## 📌 Features
- 🏗 **CNN Model** trained on ASL dataset
- 📷 **Real-time gesture recognition**
- 🖼 **Image-based classification**
- 📊 **Data preprocessing and augmentation**
- 🌍 **Deployed using Streamlit**

---

## 📥 How to Run Locally

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/AryanDhanuka10/Sign_Language_Detection.git
cd Sign_Language_Detection
```

### **2️⃣ Setup Virtual Environment**
```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate   # On Windows, use: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Application**
```bash
streamlit run app.py
```

---

## 📊 Dataset
The dataset used is an **American Sign Language (ASL) alphabet dataset** (excluding the letters 'J' and 'Z' as they require motion).

📌 **Dataset Source**: Available on **Kaggle**

🖼 **Sample Image**:
![ASL Alphabet](src/images/asl.png)

---

## 🛠 Model Architecture
The project is built using **Convolutional Neural Networks (CNNs)** to classify ASL hand gestures.

### 🔹 **Layers Used**:
- **Conv2D**: Extracts spatial features from images.
- **BatchNormalization**: Normalizes activations to improve training.
- **MaxPooling2D**: Reduces feature dimensions while preserving important information.
- **Dropout**: Prevents overfitting by randomly deactivating neurons.
- **Flatten**: Converts multidimensional tensors into vectors.
- **Dense**: Fully connected layers for classification.

📌 **Model Visualization**:
<img src="src/images/cnn.png" width="400px" />

---

## 🚀 Deployment
The project is deployed on **Streamlit**. You can access the live demo here:

🌐 **[Sign Language Detection App](https://signlanguagedetection-aryandhanuka10.streamlit.app/)**

---

## 🎯 Future Improvements
- 🎥 **Add real-time video detection**
- 🤖 **Enhance model accuracy with more training data**
- 📊 **Implement Transfer Learning using pre-trained models**
- 📱 **Deploy as a mobile app**

---

## 🤝 Contributing
Contributions are welcome! If you find a bug or want to improve the model, feel free to submit a **Pull Request**.

---

## 📜 License
This project is **open-source** and available under the [MIT License](LICENSE).

