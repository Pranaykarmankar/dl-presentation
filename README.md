<p align="center">
  <strong>⚡ WeldVision AI — Source Code</strong>
</p>

<h3 align="center">Intelligent Welding Defect Detection System</h3>

<p align="center">
  <em>Dual-engine deep learning pipeline combining CNN classification and YOLOv8 object detection for automated weld quality inspection.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/YOLOv8-ONNX-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/PyQt5-5.15+-41CD52?logo=qt&logoColor=white" alt="PyQt5"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Module 1: CNN Classifier](#-module-1-cnn-classifier-streamlit-web-app)
- [Module 2: YOLOv8 Inspector](#-module-2-yolov8-inspector-pyqt5-desktop-app)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Screenshots](#-screenshots)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

**WeldVision AI** is a comprehensive welding defect detection system built for real-time quality inspection in manufacturing environments. It combines two complementary deep learning approaches:

| Approach | Purpose | Interface |
|----------|---------|-----------|
| **CNN Classifier** | Quick pass/fail classification with confidence scores | Streamlit Web App |
| **YOLOv8 Inspector** | Precise defect localization with bounding boxes + PDF reports | PyQt5 Desktop App |

The system is designed for industrial use cases including:
- **Manufacturing QC** — Rapid weld quality assessment on production lines
- **Training & Education** — Visual demonstration of deep learning in NDT (Non-Destructive Testing)
- **Field Inspection** — Portable inspection via Raspberry Pi kiosk mode
- **Reporting** — Automated PDF report generation with annotated images

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────┐
                    │      WeldVision AI Hub       │
                    │     (Streamlit Landing)      │
                    └──────────┬──────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        ┌────────▼────────┐        ┌─────────▼────────┐
        │  CNN Classifier │        │  YOLOv8 Inspector │
        │   (Streamlit)   │        │     (PyQt5)       │
        └────────┬────────┘        └─────────┬─────────┘
                 │                           │
        ┌────────▼────────┐        ┌─────────▼─────────┐
        │  Keras .h5      │        │  ONNX Runtime     │
        │  Sequential CNN │        │  YOLOv8 Model     │
        │  150×150 input  │        │  640×640 input     │
        └────────┬────────┘        └─────────┬─────────┘
                 │                           │
        ┌────────▼────────┐        ┌─────────▼─────────┐
        │  5 Classes:     │        │  4 Defect Types:  │
        │  • Crack        │        │  • Crack          │
        │  • Good Weld    │        │  • Excess Reinf.  │
        │  • Porosity     │        │  • Porosity       │
        │  • Slag Incl.   │        │  • Spatters       │
        │  • Spatter      │        │                   │
        └─────────────────┘        └─────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │  PDF Report Gen  │
                                    │  + Annotated Img │
                                    └──────────────────┘
```

---

## 📁 Repository Structure

```
weldvision-source-code/
│
├── README.md                              ← You are here
├── .gitignore
│
├── cnn/                                   ← MODULE 1: CNN Classifier
│   ├── app.py                             ← Streamlit web application (main entry)
│   ├── welding_defect_cnn_model.h5        ← Trained Keras/TensorFlow CNN model (~65 MB)
│   ├── classes.txt                        ← Class label definitions
│   ├── requirements.txt                   ← Python dependencies for CNN module
│   ├── notebookc5cb3f14ca.ipynb           ← Jupyter notebook (training pipeline)
│   ├── extracted_code.py                  ← Python code extracted from notebook
│   ├── extract.py                         ← Utility: extract code from notebook
│   ├── extract_classes.py                 ← Utility: extract class names from model
│   └── raw_outputs.txt                    ← Raw training logs and outputs
│
├── yolo/                                  ← MODULE 2: YOLOv8 Inspector
│   ├── main.py                            ← PyQt5 application entry point
│   ├── requirements.txt                   ← Python dependencies for YOLO module
│   │
│   ├── controller/
│   │   ├── __init__.py
│   │   └── app_controller.py              ← Navigation controller + data flow
│   │
│   ├── ui/                                ← User interface pages
│   │   ├── __init__.py
│   │   ├── topbar.py                      ← Persistent top navigation bar
│   │   ├── page_index.py                  ← Home / landing screen
│   │   ├── page_new_scan.py               ← 4-step specimen details wizard
│   │   ├── page_camera.py                 ← Live camera feed + image capture
│   │   ├── page_analysis.py               ← AI detection results display
│   │   ├── page_history.py                ← Scan history list
│   │   └── page_report_viewer.py          ← PDF report viewer
│   │
│   ├── services/                          ← Business logic layer
│   │   ├── __init__.py
│   │   ├── ai_service.py                  ← YOLOv8 ONNX inference engine
│   │   ├── camera_service.py              ← OpenCV camera management
│   │   └── report_service.py              ← PDF report generation
│   │
│   ├── utils/                             ← Shared utilities
│   │   ├── __init__.py
│   │   ├── constants.py                   ← Theme, colors, paths, storage helpers
│   │   └── widgets.py                     ← Reusable styled Qt widget factories
│   │
│   ├── model/
│   │   └── best.onnx                      ← Trained YOLOv8 ONNX model
│   │
│   ├── static/                            ← Logo and brand assets
│   │   ├── sigmandt.png
│   │   └── sigmandt.avif
│   │
│   ├── scans/                             ← Saved capture images (auto-created)
│   ├── reports/                           ← Generated PDF reports (auto-created)
│   ├── scans.json                         ← Scan records database (JSON)
│   ├── test_model.py                      ← Model testing / validation script
│   └── welding_report.py                  ← Report template and generation logic
```

---

## 🧠 Module 1: CNN Classifier (Streamlit Web App)

A browser-based welding defect classifier powered by a custom-trained Keras/TensorFlow Convolutional Neural Network.

### Features

- **Image Upload** — Drag-and-drop or browse for weld images (JPG, PNG)
- **Webcam Capture** — Real-time image capture directly from browser
- **Interactive Cropping** — Select a region of interest before analysis using `streamlit-cropper`
- **Confidence Scores** — Full probability distribution across all 5 classes
- **Inference Timing** — Per-image inference time in milliseconds
- **Mobile Responsive** — Optimized UI for smartphones and tablets
- **Premium UI** — Glassmorphism design with gradient backgrounds, hover animations, and Inter font

### Detection Classes

| Class | Description |
|-------|-------------|
| **Crack** | Linear discontinuities in the weld bead |
| **Good Weld** | Acceptable weld with no visible defects |
| **Porosity** | Gas pockets trapped during solidification |
| **Slag Inclusion** | Non-metallic material trapped in weld metal |
| **Spatter** | Metal droplets expelled during welding |

### How It Works

1. User uploads an image or captures via webcam
2. Image is optionally cropped to the region of interest
3. Image is resized to **150×150 pixels** and normalized
4. Forward pass through the Keras Sequential CNN model
5. Softmax output returns probabilities for all 5 classes
6. Results displayed with confidence bars and pass/fail status

---

## 🎯 Module 2: YOLOv8 Inspector (PyQt5 Desktop App)

A full-featured desktop inspection application with a multi-step workflow, live camera support, AI-powered defect detection with bounding boxes, and automated PDF report generation.

### Features

- **4-Step Specimen Wizard** — Structured data entry (ID, operator, material, dimensions, process)
- **Live Camera Feed** — Real-time USB/CSI camera preview with OpenCV
- **Image Upload** — Alternative to camera for pre-captured images
- **YOLOv8 Detection** — Object detection with bounding box localization
- **Non-Maximum Suppression (NMS)** — Eliminates duplicate detections
- **Letterbox Preprocessing** — Maintains aspect ratio for accurate inference
- **PDF Report Generation** — Professional inspection reports with annotated images
- **Scan History** — Browse and review past inspection records
- **Kiosk Mode** — Auto-fullscreen on Linux/Raspberry Pi
- **Simulation Fallback** — UI works without the model for development/demo

### Detection Classes

| Class | Description |
|-------|-------------|
| **Crack** | Fractures in the weld or heat-affected zone |
| **Excess Reinforcement** | Excessive weld metal above the base surface |
| **Porosity** | Cavities caused by gas entrapment |
| **Spatters** | Unwanted metal droplets around the weld area |

### Inspection Workflow

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Specimen  │───▶│ 2. Capture /  │───▶│ 3. AI        │───▶│ 4. PDF       │
│    Details   │    │    Upload     │    │    Analysis   │    │    Report    │
│              │    │    Image      │    │    (YOLOv8)   │    │    Output   │
└──────────────┘    └───────────────┘    └──────────────┘    └──────────────┘
  • Sample ID         • USB Webcam        • Bounding boxes    • Annotated image
  • Operator          • CSI Camera        • Confidence %      • Defect summary
  • Material          • File upload       • Severity rating   • Specimen info
  • Dimensions        • Crop support      • Inference time    • Timestamp
  • Process info                                               • Pass/Fail
```

### Severity Rating System

| Defect Count | Severity | Action |
|-------------|----------|--------|
| 0 | ✅ Pass | No defects detected |
| 1–2 | ⚠️ Minor | Review recommended |
| 3–5 | 🟠 Moderate | Repair likely needed |
| 6+ | 🔴 Severe | Immediate attention required |

---

## 🚀 Installation & Setup

### Prerequisites

- Python **3.8+**
- pip (Python package manager)
- Git

### Clone the Repository

```bash
git clone https://github.com/Pranaykarmankar/weldvision-source-code.git
cd weldvision-source-code
```

### CNN Module Setup

```bash
cd cnn
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### YOLO Module Setup

```bash
cd yolo
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** On Raspberry Pi or Linux with CUDA, replace `onnxruntime` with `onnxruntime-gpu` for GPU acceleration.

---

## 💻 Usage

### Run the CNN Classifier (Web)

```bash
cd cnn
streamlit run app.py
```

The Streamlit app opens at `http://localhost:8501`. Choose an input method (upload or webcam), optionally crop the image, and click **Analyze**.

### Run the YOLOv8 Inspector (Desktop)

```bash
cd yolo
python main.py
```

The PyQt5 desktop window launches. Follow the 4-step wizard to enter specimen details, capture/upload an image, run AI analysis, and generate a PDF report.

### Run via Unified Launcher

The CNN module's `app.py` also serves as a **unified launcher** — it provides a model selection landing page where you can:
1. Launch the **CNN Classifier** inline (Streamlit)
2. Launch the **YOLOv8 Inspector** as a separate desktop process

---

## 📊 Model Details

### CNN Model (`welding_defect_cnn_model.h5`)

| Property | Value |
|----------|-------|
| **Framework** | TensorFlow / Keras |
| **Architecture** | Sequential CNN |
| **Input Size** | 150 × 150 × 3 (RGB) |
| **Output** | 5-class softmax |
| **File Size** | ~65 MB |
| **Format** | HDF5 (`.h5`) |
| **Training** | Custom dataset (see `notebookc5cb3f14ca.ipynb`) |

### YOLOv8 Model (`best.onnx`)

| Property | Value |
|----------|-------|
| **Framework** | Ultralytics YOLOv8 → ONNX Runtime |
| **Architecture** | YOLOv8 |
| **Input Size** | 640 × 640 × 3 (RGB, letterboxed) |
| **Output** | Bounding boxes + class scores (8400 candidates) |
| **Classes** | 4 (crack, excess_reinforcement, porosity, spatters) |
| **Confidence Threshold** | 0.45 |
| **NMS IoU Threshold** | 0.45 |
| **Format** | ONNX |
| **Preprocessing** | Letterbox resize + border padding (114, 114, 114) |

---

## 🖥️ Screenshots

> _Run the application to see the premium UI with glassmorphism effects, gradient backgrounds, and responsive design._

**CNN Classifier Features:**
- Hero landing page with model selection cards
- Interactive image cropping with drag handles
- Confidence bar visualization
- Pass/Fail result cards with color-coded borders

**YOLOv8 Inspector Features:**
- Multi-step specimen form with dark theme
- Live camera preview window
- Bounding box overlay on detected defects
- Professional PDF reports with annotated images

---

## 🍇 Deployment

### Raspberry Pi (Kiosk Mode)

The YOLOv8 Inspector is designed for Raspberry Pi 3 B+ kiosk deployment:

```bash
# CSI Camera setup
sudo modprobe bcm2835-v4l2

# Set model path (optional)
export WELD_MODEL_PATH=/home/pi/Desktop/Weld-Inspection/model/best.onnx

# Launch in kiosk mode (auto-fullscreen on Linux)
cd yolo
python3 main.py
```

### Streamlit Cloud

The CNN module can be deployed to [Streamlit Cloud](https://streamlit.io/cloud):

1. Push the `cnn/` directory contents to a GitHub repo
2. Connect your repo on Streamlit Cloud
3. Set the main file to `app.py`
4. The `.h5` model file must be included in the repo (or use Git LFS)

### Docker (Optional)

```dockerfile
# CNN Module
FROM python:3.10-slim
WORKDIR /app
COPY cnn/ .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **CNN Training** | TensorFlow 2.x, Keras, NumPy |
| **CNN Interface** | Streamlit, streamlit-cropper, Pillow |
| **YOLO Inference** | ONNX Runtime, OpenCV, NumPy |
| **YOLO Interface** | PyQt5, QStackedWidget (page navigation) |
| **Image Processing** | OpenCV (resize, letterbox, NMS, annotations) |
| **Report Generation** | Custom PDF builder with annotated images |
| **Camera** | OpenCV VideoCapture (USB + CSI) |
| **Design** | Custom CSS (glassmorphism, gradients, Inter font) |
| **Deployment** | Raspberry Pi, Streamlit Cloud, Docker |

---

## 👨‍💻 Author

**Pranay Karmankar**

- GitHub: [@Pranaykarmankar](https://github.com/Pranaykarmankar)

---

## 📄 License

This project is for educational and presentation purposes.

---

<p align="center">
  <em>Built with ❤️ using TensorFlow, YOLOv8, Streamlit, and PyQt5</em>
</p>
