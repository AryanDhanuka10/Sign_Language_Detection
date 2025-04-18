import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained sign language model
model_path = "/home/aryan-dhanuka/DTI Project/artifacts/model/sign_language_model.h5"
model = load_model(model_path)

# Mapping from model output to sign language labels
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Preprocess the frame captured from the webcam
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Resize image
    resized_frame = np.expand_dims(resized_frame, axis=-1)  # Add channel dimension
    resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    resized_frame = resized_frame / 255.0  # Normalize
    return resized_frame

# Streamlit UI
st.set_page_config(page_title="Sign Language Detection", layout="wide")

st.title("🖐️ Real-time Sign Language Detection")
st.write("Make a sign in front of your webcam to detect the sign language letter!")

# Sidebar for Controls
st.sidebar.header("🔧 Settings")
run_webcam = st.sidebar.toggle("Enable Webcam", value=True)

# Webcam Capture
if run_webcam:
    cap = cv2.VideoCapture(0)  # Start webcam
    stframe = st.empty()  # Placeholder for video feed

    while run_webcam:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            st.sidebar.error("❌ Failed to capture image")
            break

        # Preprocess the captured frame
        processed_frame = preprocess_frame(frame)

        # Model Prediction
        prediction = model.predict(processed_frame)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Assign label
        predicted_label = class_labels.get(predicted_class, "Unknown")

        # Draw bounding box & prediction text
        cv2.rectangle(frame, (10, 10), (200, 50), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {predicted_label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB and display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()  # Release webcam when stopped
    st.sidebar.write("✅ Webcam Stopped")

