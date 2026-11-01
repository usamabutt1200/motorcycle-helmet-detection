import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

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
    width: 100%;
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
    st.error("Critical Error: Failed to initialize helmet_model.pt. Verify its presence in the repository.")
    st.stop()

# ---------------- SESSION STATE INITIALIZATION ----------------
if "video_data_buffer" not in st.session_state:
    st.session_state.video_data_buffer = None
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
    <h1>🪖 HELMET AI</h1>
    <h3>Motorcycle Helmet Detection System</h3>
    <p class="small-text">
        Intelligent AI-powered helmet detection platform for real-time safety enforcement.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.markdown("# 🪖 HELMET AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION PANEL",
    ["Dashboard", "Image Detection", "Video Detection", "Live Detection"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by YOLOv8 Ecosystem")

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("System Dashboard")
    st.write("Welcome to the Helmet AI Analytical Platform. Select a module from the sidebar to begin processing.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h2>📷 Image Mode</h2><p>Execute precision detection on uploaded static images.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h2>🎥 Video Mode</h2><p>Process dynamic recordings with automated high-speed inference matching.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h2>📸 Snapshot Mode</h2><p>Capture instant image streams from your camera input.</p></div>', unsafe_allow_html=True)

# ---------------- IMAGE DETECTION ----------------
elif page == "Image Detection":
    st.title("📷 Static Image Analytics")

    uploaded_image = st.file_uploader("Upload target asset source", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")

        if st.button("Execute Neural Inference"):
            with st.spinner("Analyzing structural visual fields..."):
                results = model.predict(np.array(image), conf=0.35, imgsz=640, verbose=False)
                result = results[0]
                annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Source Asset")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Inference Result Map")
                st.image(annotated, use_container_width=True)

            st.subheader("Object Metrics Log")
            if result.boxes is None or len(result.boxes) == 0:
                st.info("Zero violations detected. Target area compliant.")
            else:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0]) * 100
                    label = model.names[class_id]
                    st.write(f"Class Signature: **{label}** — Inference Confidence: {confidence:.2f}%")

# ---------------- DYNAMIC VIDEO DETECTION ----------------
elif page == "Video Detection":
    st.title("🎥 Dynamic Video Analytics")

    uploaded_video = st.file_uploader("Upload dynamic video stream sequence", type=["mp4", "avi", "mov", "mkv"])

    # Clear cached bytes if a brand new video file is uploaded
    if uploaded_video:
        current_file_key = f"video_{uploaded_video.name}_{uploaded_video.size}"
        if st.session_state.last_processed_file != current_file_key:
            st.session_state.video_data_buffer = None
            st.session_state.last_processed_file = current_file_key

    # Execution UI View Context Container
    processing_placeholder = st.empty()

    if uploaded_video and st.session_state.video_data_buffer is None:
        # Wrap execution UI layout items inside this block container
        with processing_placeholder.container():
            if st.button("Initialize High-Speed Pipeline Execution"):
                with st.spinner("Neural engine processing frames dynamically... Please maintain connection."):
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_file:
                        input_file.write(uploaded_video.getvalue())
                        input_path = input_file.name

                    cap = cv2.VideoCapture(input_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)
                    frame_count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Speed optimization: Alternating frames + lightweight inference resolution
                        if frame_count % 2 == 0:
                            results = model.predict(frame, conf=0.35, imgsz=320, verbose=False)
                            annotated_frame = results[0].plot()
                        else:
                            annotated_frame = frame

                        writer.write(annotated_frame)
                        frame_count += 1

                        if total_frames > 0:
                            progress_bar.progress(min(frame_count / total_frames, 1.0))

                    cap.release()
                    writer.release()

                    # Load fully processed tracking data into state memory safely
                    with open(raw_output_path, "rb") as raw_file:
                        st.session_state.video_data_buffer = raw_file.read()
                
                # Triggers layout rerun immediately after processing is finished
                st.rerun()

    # Success Dashboard Output Layout (Appears after cleanup)
    if st.session_state.video_data_buffer is not None:
        # Completely collapse and destroy progress bars, spinners, and initialization toggles
        processing_placeholder.empty() 
        
        st.success("Analysis Pipeline Complete! The tracking process is successful.")
        st.markdown('<div class="card"><h3>🎯 Ready to Export Asset File</h3></div>', unsafe_allow_html=True)
        
        # Immediate memory stream download interaction loop block
        st.download_button(
            label="⬇️ Download Processed Output File",
            data=st.session_state.video_data_buffer,
            file_name="helmet_detection_processed.mp4",
            mime="video/mp4",
            key="instant_download_node"
        )

# ---------------- STATIC SNAPSHOT LIVE DETECTION ----------------
elif page == "Live Detection":
    st.title("📸 Precision Snapshot Analytics")
    st.write("Native single-frame capture mode to guarantee consistent application runtime stability.")
    st.markdown("---")

    img_file_buffer = st.camera_input("Capture view matrix frame target")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("Parsing localized feature mappings..."):
            # Conf=0.10 karne se model ko majbooran box draw karna parega thode se shaq par bhi
            results = model.predict(cv2_img, conf=0.10, imgsz=640, verbose=False)
            result = results[0] if isinstance(results, list) else results
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            
        st.subheader("🎯 Snapshot Verification Analysis")
        st.image(annotated, use_container_width=True)
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                confidence = (float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)) * 100
                label = model.names[class_id]
                st.error(f"Incident Event Flagged: **{label}** verified with a threshold of {confidence:.2f}%")
        else:
            st.success("Target environment assessment clear. No infractions logged.")

