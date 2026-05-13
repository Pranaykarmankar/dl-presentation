import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os, time, subprocess, sys

st.set_page_config(page_title="WeldVision AI", page_icon="⚡", layout="wide")

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

.hero-title {
    text-align: center; padding: 2rem 0 0.5rem;
    background: linear-gradient(135deg, #00d2ff, #7b2ff7, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.8rem; font-weight: 800; letter-spacing: -1px;
}
.hero-sub {
    text-align: center; color: #8892b0; font-size: 1.1rem;
    margin-bottom: 2rem; font-weight: 300;
}
.model-card {
    background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 2rem; text-align: center;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1); min-height: 280px;
}
.model-card:hover { border-color: rgba(0,210,255,0.4); transform: translateY(-8px);
    box-shadow: 0 20px 60px rgba(0,210,255,0.15); }
.model-card h3 { color: #e6f1ff; font-size: 1.5rem; margin: 1rem 0 0.5rem; font-weight: 700; }
.model-card p { color: #8892b0; font-size: 0.9rem; line-height: 1.6; }
.model-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin: 0.25rem;
}
.badge-purple { background: rgba(123,47,247,0.2); color: #b388ff; }
.badge-cyan { background: rgba(0,210,255,0.2); color: #00d2ff; }
.glass-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem; margin: 0.5rem 0;
    backdrop-filter: blur(20px);
}
.result-box {
    background: rgba(255,255,255,0.05); border-radius: 12px;
    padding: 1.2rem; margin: 0.5rem 0; border-left: 4px solid;
}
.result-pass { border-color: #22c55e; }
.result-fail { border-color: #ef4444; }
.stat-value { font-size: 2rem; font-weight: 800; color: #e6f1ff; }
.stat-label { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
h1,h2,h3,h4,h5,h6,p,span,div,label { color: #ccd6f6 !important; }
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.6rem 2rem !important; font-weight: 600 !important;
    transition: all 0.3s !important;
}
.stButton > button:hover { transform: scale(1.05) !important; box-shadow: 0 8px 30px rgba(123,47,247,0.4) !important; }
.stFileUploader { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }
div[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; padding-top: 1rem; }
    .hero-sub { font-size: 0.95rem; margin-bottom: 1.5rem; padding: 0 1rem; }
    .model-card { min-height: auto; padding: 1.5rem 1rem; margin-bottom: 1rem; }
    .model-icon { font-size: 2.5rem; }
    .model-card h3 { font-size: 1.3rem; }
    .stat-value { font-size: 1.5rem; }
    .stButton > button { padding: 0.8rem 1rem !important; }
    .glass-card { padding: 1rem; }
    h3 { font-size: 1.4rem; }
    /* Hide streamlit empty columns on mobile if they stack as blank space */
    [data-testid="column"]:empty { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ── CNN Model Loader ─────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    try:
        import tensorflow as tf
        path = os.path.join(os.path.dirname(__file__), 'welding_defect_cnn_model.h5')
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"CNN load error: {e}")
    return None

CNN_CLASSES = ['Crack', 'Good Weld', 'Porosity', 'Slag Inclusion', 'Spatter']

def predict_cnn(image, model):
    img = np.array(image)
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (150, 150)).astype('float32')
    img = np.expand_dims(img, axis=0)
    t0 = time.perf_counter()
    preds = model.predict(img, verbose=0)[0]
    elapsed = (time.perf_counter() - t0) * 1000
    classes = CNN_CLASSES if len(preds) == len(CNN_CLASSES) else [f"Class {i}" for i in range(len(preds))]
    top_idx = np.argmax(preds)
    results = {classes[i]: float(preds[i]) for i in range(len(preds))}
    return classes[top_idx], float(preds[top_idx]), results, elapsed

# ── SESSION STATE ────────────────────────────────────────────────────────────
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = None

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚡ WeldVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Intelligent Welding Defect Detection — CNN & YOLO Powered</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.model_choice is None:
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>🎯 Choose Your Detection Engine</h3>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""<div class="model-card">
            <div class="model-icon">🧠</div>
            <h3>CNN Classifier</h3>
            <p>TensorFlow/Keras CNN trained for <b>image classification</b>. Upload an image and get class predictions with confidence scores.</p>
            <span class="badge badge-purple">TensorFlow</span>
            <span class="badge badge-purple">150×150</span>
            <span class="badge badge-purple">5 Classes</span>
            <p style="margin-top:1rem;font-size:0.85rem;color:#a78bfa !important;">
            ✦ Upload / Webcam → Class Prediction<br>
            ✦ Confidence Distribution<br>
            ✦ Runs inside Streamlit</p>
        </div>""", unsafe_allow_html=True)
        if st.button("▶  Launch CNN Classifier", key="btn_cnn", use_container_width=True):
            st.session_state.model_choice = "CNN"
            st.rerun()

    with col_b:
        st.markdown("""<div class="model-card">
            <div class="model-icon">🎯</div>
            <h3>YOLOv8 Inspector</h3>
            <p>Full inspection pipeline with <b>bounding box detection</b>. Enter specimen details → capture/upload image → AI analysis → PDF report.</p>
            <span class="badge badge-cyan">ONNX Runtime</span>
            <span class="badge badge-cyan">640×640</span>
            <span class="badge badge-cyan">4 Classes</span>
            <p style="margin-top:1rem;font-size:0.85rem;color:#67e8f9 !important;">
            ✦ 4-Step Specimen Form<br>
            ✦ Camera / Upload → Bounding Boxes<br>
            ✦ PDF Report Generation</p>
        </div>""", unsafe_allow_html=True)
        if st.button("▶  Launch YOLO Inspector", key="btn_yolo", use_container_width=True):
            st.session_state.model_choice = "YOLO"
            st.rerun()

    # Comparison table
    st.markdown("---")
    st.markdown("#### 📊 Model Comparison")
    st.table({
        "Feature": ["Architecture", "Input Size", "Output Type", "Classes", "Best For", "Interface"],
        "CNN Classifier": ["Keras Sequential", "150×150", "Class Probabilities", "5 (incl. Good Weld)", "Quick Classification", "Streamlit Web UI"],
        "YOLOv8 Inspector": ["YOLOv8 ONNX", "640×640", "Bounding Boxes", "4 (defect types)", "Localization + Reporting", "Desktop PyQt5 App"]
    })
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# YOLO PATH — Launch the PyQt5 desktop app
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.model_choice == "YOLO":
    st.markdown("---")
    st.markdown("### 🎯 YOLOv8 Weld Inspector")
    st.markdown("""
    <div class="glass-card">
        <h4>🚀 Launching Desktop Inspector App...</h4>
        <p>The YOLOv8 Inspector runs as a separate <b>PyQt5 desktop application</b> with its full pipeline:</p>
        <ol>
            <li><b>Identification</b> — Sample ID, operator, report details</li>
            <li><b>Specimen Info</b> — Type, joint configuration, material</li>
            <li><b>Dimensions</b> — Width × Height in mm</li>
            <li><b>Process</b> — Welding process, industry, notes</li>
            <li><b>Camera / Upload</b> — Capture or upload specimen image</li>
            <li><b>AI Analysis</b> — YOLOv8 defect detection with bounding boxes</li>
            <li><b>PDF Report</b> — Generate a professional inspection report</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Launch YOLO Inspector App", use_container_width=True, type="primary"):
            yolo_dir = os.path.join(os.path.dirname(__file__), 'dl-presentation')
            yolo_main = os.path.join(yolo_dir, 'main.py')
            if os.path.exists(yolo_main):
                subprocess.Popen([sys.executable, yolo_main], cwd=yolo_dir)
                st.success("✅ YOLO Inspector launched! Check your taskbar for the desktop window.")
            else:
                st.error(f"❌ Could not find: {yolo_main}")
    with col2:
        if st.button("🔄 Back to Model Selection", use_container_width=True):
            st.session_state.model_choice = None
            st.rerun()
    
    st.info("💡 **Tip:** The YOLO Inspector opens as a separate desktop window. You can keep this page open or switch models anytime.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CNN PATH — Full Streamlit classifier UI
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 CNN Classifier")
    st.markdown('<span class="badge badge-purple">CNN Active</span>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 Switch Model", use_container_width=True):
        st.session_state.model_choice = None
        st.rerun()
    st.markdown("---")
    input_method = st.radio("📸 Input Method", ["Upload Image", "Webcam"], index=0)
    st.markdown("---")
    st.markdown("**Detection Classes:**")
    for c in CNN_CLASSES:
        st.markdown(f"• {c}")

if input_method == "Upload Image":
    uploaded = st.file_uploader("Drop a welding image here", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        
        try:
            from streamlit_cropper import st_cropper
        except ImportError:
            st.error("Please install streamlit-cropper: `pip install streamlit-cropper`")
            st.stop()
            
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📷 Original Image")
            cropped_img = st_cropper(image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### ✂️ Cropped Preview")
            if cropped_img:
                st.image(cropped_img, use_container_width=True)
                
            if st.button("🔬 Analyze Cropped Area", type="primary", use_container_width=True):
                model = load_cnn_model()
                if model:
                    with st.spinner("Running CNN inference..."):
                        label, conf, results, ms = predict_cnn(cropped_img, model)
                    is_good = label == "Good Weld"
                    css = "result-pass" if is_good else "result-fail"
                    emoji = "✅" if is_good else "⚠️"
                    st.markdown(f'<div class="result-box {css}"><div class="stat-value">{emoji} {label}</div><div class="stat-label">Confidence: {conf:.1%} | Inference: {ms:.0f}ms</div></div>', unsafe_allow_html=True)
                    st.markdown("##### 📊 Probability Distribution")
                    for cls, prob in sorted(results.items(), key=lambda x: -x[1]):
                        st.progress(prob, text=f"{cls}: {prob:.1%}")
                else:
                    st.error("❌ CNN model file not found!")
            st.markdown('</div>', unsafe_allow_html=True)

elif input_method == "Webcam":
    st.markdown("### 📸 Webcam Capture")
    cam = st.camera_input("Take a picture to analyze")
    if cam:
        image = Image.open(cam).convert("RGB")
        model = load_cnn_model()
        if model:
            with st.spinner("Analyzing..."):
                label, conf, results, ms = predict_cnn(image, model)
            is_good = label == "Good Weld"
            css = "result-pass" if is_good else "result-fail"
            emoji = "✅" if is_good else "⚠️"
            st.markdown(f'<div class="result-box {css}"><div class="stat-value">{emoji} {label}</div><div class="stat-label">Confidence: {conf:.1%} | Inference: {ms:.0f}ms</div></div>', unsafe_allow_html=True)
            st.markdown("##### 📊 Probability Distribution")
            for cls, prob in sorted(results.items(), key=lambda x: -x[1]):
                st.progress(prob, text=f"{cls}: {prob:.1%}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#4a5568;font-size:0.8rem;">WeldVision AI • CNN & YOLOv8 Dual-Engine • Built for Presentation</div>', unsafe_allow_html=True)
