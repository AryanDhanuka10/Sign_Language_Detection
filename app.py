import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import gdown
from PIL import Image
import os

# Disable oneDNN optimizations for consistent behavior (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "-1"

# Google Drive model download if not exists
file_id = "1kf2lXMvj3kr-uFKoJR54VZWjQBdpVWcN"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "sign_language_model.h5"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)

# Class labels
class_labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '_'
}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    resized = resized.reshape(1, 64, 64, 1) / 255.0
    return resized

# Streamlit UI
st.set_page_config(page_title="Sign Language Detector", layout="centered")
st.title("🖐️ Real-Time Sign Language Detection")

# Choose mode (Webcam or Photo)
mode = st.sidebar.radio("Select Mode:", ["📸 Photo (Web Compatible)", "🎥 Live Webcam (Local & Cloud)"])

if mode == "🎥 Live Webcam (Local & Cloud)":
    # Local Webcam (OpenCV) and Cloud Webcam (camera_input)
    run_webcam = st.sidebar.button("▶ Start Webcam")
    stop_webcam = st.sidebar.button("⏹ Stop Webcam")

    if run_webcam and not stop_webcam:
        # Local webcam with OpenCV (for local environments)
        cap = cv2.VideoCapture(2)
        stframe = st.empty()
        box_start = (220, 140)
        box_end = (420, 340)

        st.info("✅ Webcam running... Click 'Stop Webcam' in the sidebar to end session.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Could not access webcam.")
                break

            frame = cv2.resize(frame, (640, 480))
            cv2.rectangle(frame, box_start, box_end, (255, 255, 255), 2)
            roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]

            # Prediction
            input_data = preprocess_frame(roi)
            prediction = model.predict(input_data)
            predicted_label = class_labels.get(np.argmax(prediction), "Unknown")

            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", width=640)

            time.sleep(0.05)
            if stop_webcam:
                break

        cap.release()
        st.sidebar.success("🛑 Webcam stopped.")
    else:
        st.info("ℹ️ Click 'Start Webcam' to begin real-time detection (works both locally and on Streamlit Cloud).")

else:  # 📸 Photo mode
    st.write("📷 Use your webcam to take a snapshot")
    img_data = st.camera_input("Capture Gesture")

    if img_data is not None:
        img = Image.open(img_data)
        input_data = preprocess_frame(np.array(img))
        prediction = model.predict(input_data)
        predicted_label = class_labels.get(np.argmax(prediction), "Unknown")

        st.image(img, caption="Captured Image", width=300)
        st.success(f"✅ Prediction: **{predicted_label}**")