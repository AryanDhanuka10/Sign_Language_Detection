import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained model
model_path = "artifacts/model/sign_language_model.h5"
model = load_model(model_path)

# Label mapping
class_labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '_'
}

# Preprocessing function
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_frame, (64, 64))
    resized = np.expand_dims(resized, axis=-1)
    resized = np.expand_dims(resized, axis=0)
    resized = resized / 255.0
    return resized

# Streamlit Setup
st.set_page_config(page_title="Sign Language Detection", layout="centered")
st.title("🖐️ Real-time Sign Language Detection")
st.write("Place your hand inside the white box in the webcam feed to detect your gesture!")

run_webcam = st.sidebar.toggle("Enable Webcam", value=True, key="enable_webcam")

if run_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    # Guide box
    box_start = (220, 140)
    box_end = (420, 340)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.sidebar.error("❌ Failed to capture frame")
            break

        # Resize for smoother processing
        frame = cv2.resize(frame, (640, 480))

        # Draw ROI box
        cv2.rectangle(frame, box_start, box_end, (255, 255, 255), 2)
        cv2.putText(frame, "Place hand here", (box_start[0], box_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Region of Interest
        roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]

        # Predict
        processed = preprocess_frame(roi)
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels.get(predicted_class, "Unknown")

        # Show prediction
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert & display in smaller window (no fullscreen)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", width=640)  # fixed width to avoid fullscreen

        # Optional FPS control to reduce CPU load
        time.sleep(0.05)

    cap.release()
    st.sidebar.success("✅ Webcam released")
