import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import tempfile
import os
from collections import Counter

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/hard-hat.png", width=80)
st.sidebar.title("PPE Kit Detection System")
st.sidebar.markdown("Detect PPE compliance in real-time or from video files using AI.")

# Main UI
st.title("🦺 PPE Kit Detection System")
st.markdown("""
Welcome to the PPE Kit Detection System!  
- Select your video source (webcam or upload a video).
- Click **Start Detection** to begin.
- Results will be displayed below with bounding boxes and class labels.
""")

source = st.selectbox("Select Video Source", ["Webcam", "Upload Video"])
uploaded_file = None
if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

run_detection = st.button("🚦 Start Detection")

classNames = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
    'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

model = YOLO("best.pt")

def process_frame(img):
    detected_classes = []
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            detected_classes.append(classNames[cls])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return img, detected_classes

if run_detection:
    st.info("Detection started. Please wait...")
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
        st.success("Using webcam for detection.")
    else:
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            st.success("Using uploaded video for detection.")
        else:
            st.warning("Please upload a video file.")
            st.stop()

    stframe = st.empty()
    summary_placeholder = st.empty()
    prev_frame_time = 0
    fps = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img, detected_classes = process_frame(img)

        new_frame_time = time.time()
        if prev_frame_time != 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            fps = int(fps)
        prev_frame_time = new_frame_time

        cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB", use_column_width=True)

        # Show summary of detections for the current frame only
        if detected_classes:
            summary = Counter(detected_classes)
            summary_text = "### Detection Summary (Current Frame)\n"
            for k, v in summary.items():
                summary_text += f"- **{k}**: {v}\n"
            summary_placeholder.markdown(summary_text)
        else:
            summary_placeholder.markdown("### Detection Summary (Current Frame)\n- No objects detected.")

    cap.release()
    if source == "Upload Video" and uploaded_file is not None:
        os.remove(tfile.name)
    st.success("Detection finished.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & YOLO | [GitHub](https://github.com/ArpitKharwade/PPE-Kit-Detection-Project)")