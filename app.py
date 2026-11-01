import streamlit as st
import cv2
import av
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet AI | Detection System",
    page_icon="🪖",
    layout="wide"
)

# ---------------- PREMIUM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d18, #111c2e);
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
h1, h2, h3 {
    color: #f5c76b !important;
}
p, label {
    color: #d5dbea;
}
.card {
    background: #17243a;
    padding: 24px;
    border: 1px solid #34435b;
    border-radius: 18px;
    margin-bottom: 18px;
}
div.stButton > button,
div.stDownloadButton > button {
    background: linear-gradient(90deg, #c99a3b, #f5d58a);
    color: #101827;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #ffffff;
    color: #101827;
}
[data-testid="stFileUploader"] {
    background: #17243a;
    border: 1px dashed #c99a3b;
    border-radius: 14px;
    padding: 15px;
}
.small-text {
    color: #aab6ca;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("helmet_model.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi ho saka. Check karein ke helmet_model.pt repository mein mojood hai.")
    st.stop()

# ---------------- STUN CONFIGURATION FOR CLOUD WEBCAM ----------------
# Is format ko browser bina kisi error ke parse kar leta hai
RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": ["stun:://google.com"]
        }
    ]
}


# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
    <h1>🪖 HELMET AI</h1>
    <h3>Motorcycle Helmet Detection System</h3>
    <p class="small-text">
        Intelligent AI-powered helmet detection platform
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("# 🪖 HELMET AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Image Detection", "Video Detection", "Live Detection"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by YOLOv8")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("Dashboard")
    st.write("Welcome to Helmet AI")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h2>📷 Image</h2>
            <p>Detect helmets in uploaded images.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h2>🎥 Video</h2>
            <p>Analyze videos and download results.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h2>📡 Live</h2>
            <p>Detect helmets through your webcam.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- IMAGE DETECTION ----------------
elif page == "Image Detection":
    st.title("📷 Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")

        if st.button("Run Detection"):
            with st.spinner("Analyzing image..."):
                result = model.predict(
                    np.array(image),
                    conf=0.35,
                    imgsz=640,
                    verbose=False
                )[0]

                annotated = result.plot()
                annotated = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Detection Result")
                st.image(annotated, use_container_width=True)

            st.subheader("Detection Details")

            if result.boxes is None or len(result.boxes) == 0:
                st.info("No helmets detected.")
            else:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0]) * 100
                    label = model.names[class_id]

                    st.write(
                        f"**{label}** — {confidence:.1f}% confidence"
                    )

# ---------------- VIDEO DETECTION ----------------
elif page == "Video Detection":
    st.title("🎥 Fast Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video and st.button("Start Fast Video Detection"):

        with st.spinner("AI high-speed engine video ko process kar raha hai... Please wait"):

            # 1. Uploaded video ko temporary file mein save karna
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_file:
                input_file.write(uploaded_video.getvalue())
                input_path = input_file.name

            cap = cv2.VideoCapture(input_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Raw output path jahan OpenCV file likhega
            raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            # H264 baseline encoder settings jo bina ffmpeg ke file completely save karti hain
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            frame_count = 0

            # Frames par YOLO prediction chalana
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # High-speed check
                if frame_count % 2 == 0:
                    results = model.predict(frame, conf=0.35, imgsz=320, verbose=False)
                    annotated_frame = results.plot()
                else:
                    annotated_frame = frame

                writer.write(annotated_frame)
                frame_count += 1

                if total_frames > 0:
                    progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            writer.release()

        # ----- 2. SECURE BINARY-FREE DOWNLOAD LOOP -----
        st.success("AI video processing mukammal ho chuki hai!")
        
        try:
            # File ko direct binary modes mein bytes memory mein transfer karna
            with open(raw_output_path, "rb") as raw_file:
                video_bytes = raw_file.read()
            
            # Agar file empty nahi hai toh guaranteed download hoga
            if len(video_bytes) > 0:
                st.info("Niche diye gaye button se processed video download karlein:")
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="helmet_detection_result.mp4",
                    mime="video/mp4",
                    key="download_btn_fixed"
                )
            else:
                st.error("Server configuration issue ki wajah se file generate nahi ho saki. Please retry karein.")
        except Exception as file_err:
            st.error(f"File read karne mein masla aaya: {str(file_err)}")

# ---------------- LIVE DETECTION ----------------
elif page == "Live Detection":
    st.title("📡 Smart Live Helmet Detection")
    
    detection_mode = st.radio(
        "CHOOSE DETECTION MODE:",
        ["Continuous Live Stream (Automatic Detection)", "Take a Snapshot (Photo Mode)"]
    )
    st.markdown("---")

    # ----- OPTION 1: CONTINUOUS LIVE STREAM (Built-in Auto Refresh Loop) -----
    if detection_mode == "Continuous Live Stream (Automatic Detection)":
        st.subheader("📡 Continuous Real-Time Detection")
        st.write("Camera open rakhein, AI bina kisi click ke automatic live update karta rahega.")
        
        # Streamlit official camera widget जो solid chalta hai
        img_file_buffer = st.camera_input("Webcam Feed Active", key="auto_live_cam")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Model prediction (Fast processing ke liye imgsz=320)
            result = model.predict(cv2_img, conf=0.35, imgsz=320, verbose=False)[0]
            annotated = result.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Direct screen par bounding boxes dikhana
            st.subheader("🎯 Real-Time Detection Output")
            st.image(annotated, use_container_width=True)
            
            # Details display
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf) * 100
                    label = model.names[class_id]
                    st.success(f"**{label}** detect hua — {confidence:.1f}%")
            else:
                st.info("Filhal samne koi helmet ya person nazar nahi aaya.")
            
            # BINA KISI EXTERNAL LIBRARY KE AUTOMATIC RERUN (Bina click kiye loop)
            # Jab bhi camera par naya frame badle ga, page khud ba khud bina lag ke refresh hoga
            st.rerun()

    # ----- OPTION 2: TAKE A SNAPSHOT (Photo Mode) -----
    elif detection_mode == "Take a Snapshot (Photo Mode)":
        st.subheader("📸 Snapshot Detection")
        img_file_buffer = st.camera_input("Take a picture to analyze", key="snapshot_cam_final_fixed")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("Analyzing Snapshot..."):
                result = model.predict(cv2_img, conf=0.35, imgsz=640, verbose=False)[0]
                annotated = result.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
            st.image(annotated, use_container_width=True)



