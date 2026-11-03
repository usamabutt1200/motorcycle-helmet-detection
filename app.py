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

/* =================================
   MAIN WEBSITE DESIGN
================================= */

.stApp {
    background: linear-gradient(135deg, #080d18, #111c2e);
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

/* HEADINGS */
h1, h2, h3 {
    color: #f5c76b !important;
}

/* TEXT */
p, label {
    color: #d5dbea;
}

/* MAIN CONTENT CARDS */
.card {
    background: #17243a;
    padding: 24px;
    border: 1px solid #34435b;
    border-radius: 18px;
    margin-bottom: 18px;
}

/* BUTTONS */
div.stButton > button,
div.stDownloadButton > button {
    background: linear-gradient(90deg, #c99a3b, #f5d58a);
    color: #101827;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
    transition: 0.2s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #ffffff;
    color: #101827;
    border: none;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #17243a;
    border: 1px dashed #c99a3b;
    border-radius: 14px;
    padding: 15px;
}

/* SMALL TEXT */
.small-text {
    color: #aab6ca;
    font-size: 14px;
}


/* =================================
   PREMIUM SIDEBAR DESIGN
================================= */

/* SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg,
        #111c30 0%,
        #0b1220 55%,
        #080d18 100%
    ) !important;

    border-right: 1px solid rgba(212, 175, 55, 0.25);
}

/* SIDEBAR SPACING */
[data-testid="stSidebar"] > div:first-child {
    padding: 25px 18px;
}

/* SIDEBAR BRAND */
[data-testid="stSidebar"] h1 {
    color: #f5c76b !important;
    font-size: 27px !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
}

/* SIDEBAR NAVIGATION HEADING */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #8fa2bd !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2px;
}

/* NAVIGATION SPACING */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 10px;
}

/* NAVIGATION BUTTONS */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #17243a !important;
    border: 1px solid #34435b !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    transition: all 0.2s ease;
}

/* HOVER EFFECT */
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #263953 !important;
    border-color: #c99a3b !important;
}

/* SELECTED PAGE */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(
        100deg,
        #c99a3b,
        #f5d58a
    ) !important;

    border: 1px solid #f5d58a !important;
    box-shadow: 0 4px 15px rgba(201, 154, 59, 0.2);
}

/* SELECTED PAGE TEXT */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #101827 !important;
    font-weight: 800 !important;
}

/* SIDEBAR NAVIGATION TEXT */
[data-testid="stSidebar"] [role="radiogroup"] label p {
    font-size: 14px;
}

/* SIDEBAR FOOTER */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    border-top: 1px solid #293b55;
    padding-top: 18px;
    margin-top: 25px;
    color: #8494aa !important;
}


/* =================================
   IMAGE & VIDEO DISPLAY
================================= */

[data-testid="stImage"] img {
    border-radius: 12px;
}

/* PROGRESS BAR */
[data-testid="stProgressBar"] > div > div {
    background-color: #d4af37;
}

/* SUCCESS MESSAGE */
[data-testid="stAlert"] {
    border-radius: 10px;
}
/* ===== CONTINUOUS ROTATING GLOW BORDER ===== */

.card,
div.stButton > button,
div.stDownloadButton > button,
[data-testid="stFileUploader"],
[data-testid="stImage"],
[data-testid="stCameraInput"] {
    position: relative;
    isolation: isolate;
    overflow: hidden;
}

/* Animated light layer */
.card::before,
div.stButton > button::before,
div.stDownloadButton > button::before,
[data-testid="stFileUploader"]::before {
    content: "";
    position: absolute;
    inset: -130%;
   background: conic-gradient(
    from 0deg,
    transparent 0deg,
    transparent 265deg,
    #c99a3b 305deg,
    #fff2bd 335deg,
    transparent 360deg
);
    animation: rotatingGlow 2.5s linear infinite;
    z-index: -2;
}

/* Inner background */
.card::after,
div.stButton > button::after,
div.stDownloadButton > button::after,
[data-testid="stFileUploader"]::after {
    content: "";
    position: absolute;
    inset: 3px;
    background: #17243a;
    border-radius: inherit;
    z-index: -1;
}

@keyframes rotatingGlow {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
/* OUTPUT MESSAGE BOX GLOW */
[data-testid="stAlert"] {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    border: 2px solid transparent;
    border-radius: 12px;
 background-color: #101A2B !important;
border: 1px solid #D4AF37 !important;
color: #F5D58A !important;
}

[data-testid="stAlert"]::before {
    content: "";
    position: absolute;
    inset: -150%;
    background: conic-gradient(
        transparent 0deg,
        transparent 265deg,
        #c99a3b 305deg,
        #fff2bd 335deg,
        transparent 360deg
    );
    animation: rotatingGlow 2.5s linear infinite;
    z-index: -1;
}

[data-testid="stAlert"] > div {
    border-radius: 10px;
}
/* ===== SEQUENTIAL LEFT SLIDE ANIMATION ===== */

@keyframes slideFromLeft {
    0% {
        opacity: 0;
        transform: translateX(-65px);
    }

    100% {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Main page elements */
.main [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
    opacity: 0;
    animation: slideFromLeft 1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

/* One by one sequence */
.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(1) {
    animation-delay: 0.1s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(2) {
    animation-delay: 0.4s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(3) {
    animation-delay: 0.7s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(4) {
    animation-delay: 1s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(5) {
    animation-delay: 1.3s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(6) {
    animation-delay: 1.6s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(7) {
    animation-delay: 1.9s;
}

.main [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(8) {
    animation-delay: 2.2s;
}

/* Accessibility */
@media (prefers-reduced-motion: reduce) {
    .main [data-testid="element-container"] {
        animation: none !important;
        opacity: 1 !important;
        transform: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("helmet_model.pt")

try:
    model = load_model()
except Exception:
    st.error(
        "Critical Error: Failed to initialize helmet_model.pt. "
        "Verify its presence in the repository."
    )
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
    st.write(
        "Welcome to the Helmet AI Analytical Platform. "
        "Select a module from the sidebar to begin processing."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="card"><h2>📷 Image Mode</h2>'
            '<p>Execute precision detection on uploaded static images.</p></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="card"><h2>🎥 Video Mode</h2>'
            '<p>Process dynamic recordings with automated high-speed inference matching.</p></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div class="card"><h2>📸 Snapshot Mode</h2>'
            '<p>Capture instant image streams from your camera input.</p></div>',
            unsafe_allow_html=True
        )

# ---------------- IMAGE DETECTION ----------------
elif page == "Image Detection":
    st.title("📷 Static Image Analytics")

    uploaded_image = st.file_uploader(
        "Upload target asset source",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")

        if st.button("Execute Neural Inference"):
            with st.spinner("Analyzing structural visual fields..."):
                result = model.predict(
                    np.array(image),
                    conf=0.25,
                    imgsz=640,
                    verbose=False
                )[0]

                annotated = cv2.cvtColor(
                    result.plot(),
                    cv2.COLOR_BGR2RGB
                )

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
                    st.write(
                        f"Class Signature: **{label}** — "
                        f"Inference Confidence: {confidence:.2f}%"
                    )

# ---------------- VIDEO DETECTION ----------------
elif page == "Video Detection":
    st.title("🎥 Dynamic Video Analytics")

    uploaded_video = st.file_uploader(
        "Upload dynamic video stream sequence",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:
        current_file_key = f"video_{uploaded_video.name}_{uploaded_video.size}"

        if st.session_state.last_processed_file != current_file_key:
            st.session_state.video_data_buffer = None
            st.session_state.last_processed_file = current_file_key

    processing_placeholder = st.empty()

    if uploaded_video and st.session_state.video_data_buffer is None:
        with processing_placeholder.container():
            if st.button("Initialize High-Speed Pipeline Execution"):
                with st.spinner(
                    "Neural engine processing frames dynamically... Please maintain connection."
                ):
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    ) as input_file:
                        input_file.write(uploaded_video.getvalue())
                        input_path = input_file.name

                    cap = cv2.VideoCapture(input_path)

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0:
                        fps = 25

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if not cap.isOpened() or width <= 0 or height <= 0:
                        st.error("Video could not be opened. Please upload a valid video.")
                        st.stop()

                    raw_output_path = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4"
                    ).name

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        raw_output_path,
                        fourcc,
                        fps,
                        (width, height)
                    )

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)
                    frame_count = 0

                    # Run detection on EVERY frame so unannotated frames
                    # are not alternated into the output video.
                    while cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            break

                        result = model.predict(
                            frame,
                            conf=0.25,
                            imgsz=640,
                            verbose=False
                        )[0]

                        annotated_frame = result.plot()
                        writer.write(annotated_frame)
                        frame_count += 1

                        if total_frames > 0:
                            progress_bar.progress(
                                min(frame_count / total_frames, 1.0)
                            )

                    cap.release()
                    writer.release()

                    with open(raw_output_path, "rb") as raw_file:
                        st.session_state.video_data_buffer = raw_file.read()

                st.rerun()

    if st.session_state.video_data_buffer is not None:
        processing_placeholder.empty()

        st.success(
            "Analysis Pipeline Complete! The tracking process is successful."
        )
        st.markdown(
            '<div class="card"><h3>🎯 Ready to Export Asset File</h3></div>',
            unsafe_allow_html=True
        )

        st.download_button(
            label="⬇️ Download Processed Output File",
            data=st.session_state.video_data_buffer,
            file_name="helmet_detection_processed.mp4",
            mime="video/mp4",
            key="instant_download_node_fixed_final"
        )

# ---------------- LIVE DETECTION ----------------
elif page == "Live Detection":
    st.title("📸 Precision Snapshot Analytics")
    st.write("Capture an image to detect helmet violations.")
    st.markdown("---")

    img_file_buffer = st.camera_input("Capture image")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()

        cv2_img = cv2.imdecode(
            np.frombuffer(bytes_data, np.uint8),
            cv2.IMREAD_COLOR
        )

        with st.spinner("Detecting helmets..."):
            result = model.predict(
                cv2_img,
                conf=0.35,
                imgsz=640,
                verbose=False
            )[0]

            annotated = cv2.cvtColor(
                result.plot(),
                cv2.COLOR_BGR2RGB
            )

        st.subheader("🎯 Detection Results")
        st.image(annotated, use_container_width=True)

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0]) * 100
                label = model.names[class_id]

                st.success(
                    f"Detected: {label} — {confidence:.2f}%"
                )
        else:
            st.warning("No helmet-related objects detected.")
