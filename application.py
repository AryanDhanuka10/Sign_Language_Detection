import streamlit as st
import cv2
import numpy as np
from src.utils.load_model import load_saved_model
from src.utils.prediction import predict_letter
from PIL import Image

# Load the model
model = load_saved_model('3_sl_aug_model.keras')

st.title("Sign Language Recognition App")
st.write("Upload an image or capture from webcam to recognize the letter.")

# Initialize session state for the webcam
if "run" not in st.session_state:
    st.session_state["run"] = False

# Start/Stop button
if st.button("Start Camera", key="start_camera"):
    st.session_state["run"] = True

if st.button("Stop Camera", key="stop_camera"):
    st.session_state["run"] = False

frame_window = st.empty()

if st.session_state["run"]:
    cap = cv2.VideoCapture(0)

    while st.session_state["run"]:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break

        # Define Region of Interest (ROI)
        roi = frame[40:300, 50:350]
        cv2.rectangle(frame, (50, 40), (350, 300), (255, 0, 0), 4)

        # Convert to grayscale and resize
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Predict the letter
        result = predict_letter(model, roi_resized)

        # Display result
        cv2.putText(frame, result, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        # Convert to RGB and show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()
    cv2.destroyAllWindows()

st.write("Camera stopped.")
