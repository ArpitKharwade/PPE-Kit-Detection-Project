

import streamlit as st
from ultralytics import YOLO
import cv2, cvzone, math, tempfile, os, time
from collections import Counter
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ---------- UI CONFIG ---------
st.set_page_config(page_title="PPE-Kit Detection System", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
body {background-color: #0E1117;}
.css-1d391kg, .css-1adrfps {background: #10151B !important;}
.sidebar .sidebar-content {background-color: #0D0F13 !important;}
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg,#2E77FF,#4BD5EE);
    border: none;
    color: white;
    font-size: 18px;
    padding: .6rem 1.2rem;
}
.stButton>button:hover {transform: scale(1.03);}
.dataCard {
    padding:15px;
    border-radius:10px;
    background:rgba(255,255,255,0.05);
    color:white;
}
h2, h3, h4, p, label {color:white !important;}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_model(): return YOLO("best.pt")
model = load_model()

classNames = [
    'Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest',
    'Person','Safety Cone','Safety Vest','machinery','vehicle'
]

# ---------- SIDEBAR ----------
st.sidebar.image("https://img.icons8.com/color/96/worker-male--v1.png", width=120)
st.sidebar.markdown("<h2>PPE-Kit Detection System</h2>", unsafe_allow_html=True)
source = st.sidebar.radio("🎥 Input Source", ["Live Webcam", "Upload Video"])
st.sidebar.info("Detect PPE compliance in real-time.")

# ---------- TURN SERVER CONFIG ----------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        { "urls": ["stun:stun.l.google.com:19302"] },
        {
            "urls": "turn:global.relay.metered.ca:80",
            "username": "openai",
            "credential": "openai"
        }
    ]
})

# ---------- YOLO DETECTION FUNCTION ----------
def process_frame(img):
    detected = []
    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]),2)
            detected.append(classNames[cls])
            cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1))
            cvzone.putTextRect(img,f"{classNames[cls]} {conf}",(x1,y1-5),scale=1,thickness=1)
    return img, detected

class VideoProcessor(VideoProcessorBase):
    def __init__(self): self.summary=""
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, cls = process_frame(img)
        if cls:
            txt="<br>".join([f"✔️ <b>{c}</b>: {n}" for c,n in Counter(cls).items()])
            self.summary=f"### 📝 Detection Summary<br>{txt}"
        else:
            self.summary="### 🚫 No PPE Detected"
        return frame.from_ndarray(img, format="bgr24")

# ---------- MAIN HEADER ----------
st.markdown("<h1 style='text-align:center;'>PPE-Kit Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Monitor worker safety with YOLOv8 in real-time 🚧</p>", unsafe_allow_html=True)
st.write("")

# ---------- LIVE WEBCAM MODE ----------
if source == "Live Webcam":
    st.subheader("📡 Live Camera Feed")
    
    ctx = webrtc_streamer(
        key="ppe-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_processor_factory=VideoProcessor
    )

    if ctx and ctx.state.playing:
        summary_box = st.empty()
        while True:
            if ctx.video_processor:
                summary_box.markdown(ctx.video_processor.summary, unsafe_allow_html=True)
            time.sleep(1)

# ---------- VIDEO UPLOAD MODE ----------
else:
    st.subheader("🎬 Upload Video for Detection")
    video = st.file_uploader("Upload a video", type=["mp4","avi","mov"])

    if video and st.button("🚦 Start Detection"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe, stats = st.empty(), st.empty()
        prev = 0
        
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break

            img, cls = process_frame(img)
            now = time.time()
            fps = int(1/(now-prev)) if prev else 0; prev = now

            cvzone.putTextRect(img, f"FPS: {fps}", (10,50))
            stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

            if cls:
                txt = "<br>".join([f"✔️ <b>{c}</b>: {n}" for c,n in Counter(cls).items()])
                stats.markdown(f"### 📝 Detection Summary<br>{txt}", unsafe_allow_html=True)

        cap.release(); os.remove(tfile.name)
        st.success("✅ Video Processing Completed!")

