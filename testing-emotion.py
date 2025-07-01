import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image

st.title("Human Emotion Detection")
st.write("Upload Image or Use Webcam for Real-Time Emotion Detection")

def analyze_img(image):
    analysis = DeepFace.analyze(image, actions=["emotion"], enforce_detection=False)
    return analysis[0]["emotion"]

upload_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if upload_file is not None:
    img = Image.open(upload_file)
    img_np = np.array(img)
    st.image(img_np, channels="RGB")
    emotion_scores = analyze_img(img_np)
    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    st.write(f"Detected Emotion: {detected_emotion}")

use_webcam = st.checkbox("Use Webcam for Real-Time Emotion Detection")

if use_webcam:
    st.write("Starting Webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        st.write("Webcam is on. Press 'q' to stop.")
        stframe = st.empty()  

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                emotion_scores = analyze_img(frame_rgb)
                detected_emotion = max(emotion_scores, key=emotion_scores.get)

                text = f"Emotion: {detected_emotion}"
                cv2.putText(frame_rgb, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                text = "No face detected"
                cv2.putText(frame_rgb, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            stframe.image(frame_rgb, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()