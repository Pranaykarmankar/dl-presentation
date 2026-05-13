# Deep Learning — Welding Defect Detection

This repository contains two deep learning approaches for welding defect detection, organized into separate modules.

---

## Repository Structure

```
dl-presentation/
│
├── README.md
│
├── cnn/                               ← CNN-based Classifier (Streamlit)
│   ├── app.py                         ← Streamlit web app for defect classification
│   ├── welding_defect_cnn_model.h5    ← Trained Keras/TF CNN model
│   ├── classes.txt                    ← Class labels for the model
│   ├── requirements.txt              ← Python dependencies for CNN app
│   ├── notebookc5cb3f14ca.ipynb       ← Training notebook
│   ├── extracted_code.py              ← Code extracted from notebook
│   ├── extract.py                     ← Extraction utility
│   ├── extract_classes.py             ← Class extraction utility
│   └── raw_outputs.txt               ← Raw training outputs
│
├── yolo/                              ← YOLOv8-based Inspector (PyQt5 Desktop)
│   ├── main.py                        ← Entry point — run this
│   ├── controller/
│   │   └── app_controller.py          ← Navigation + data flow between pages
│   ├── ui/
│   │   ├── topbar.py                  ← Persistent top bar
│   │   ├── page_index.py              ← Home screen
│   │   ├── page_new_scan.py           ← 4-step specimen details wizard
│   │   ├── page_camera.py            ← Live camera + capture
│   │   ├── page_history.py            ← Scrollable scan history list
│   │   ├── page_analysis.py          ← ML results page
│   │   └── page_report_viewer.py     ← Report viewer
│   ├── services/
│   │   ├── camera_service.py          ← OpenCV camera operations
│   │   ├── ai_service.py             ← YOLOv8 ONNX inference
│   │   └── report_service.py         ← PDF report generation
│   ├── utils/
│   │   ├── constants.py               ← Palette, paths, storage helpers
│   │   └── widgets.py                 ← Reusable styled Qt widgets
│   ├── model/
│   │   └── best.onnx                 ← Trained YOLOv8 ONNX model
│   ├── static/                        ← Logo assets
│   ├── scans/                         ← Saved captures
│   ├── reports/                       ← Generated PDF reports
│   ├── scans.json                     ← Scan records
│   ├── requirements.txt              ← Python dependencies for YOLO app
│   ├── test_model.py                 ← Model testing script
│   └── welding_report.py            ← Report generation logic
```

---

## 1. CNN Classifier (Streamlit)

A web-based welding defect classifier using a custom-trained Keras/TensorFlow CNN model.

### Quick Start

```bash
cd cnn
pip install -r requirements.txt
streamlit run app.py
```

### Features
- Upload or capture weld images
- Real-time defect classification
- Image cropping for region-of-interest selection
- Mobile-optimized responsive UI

---

## 2. YOLOv8 Inspector (PyQt5 Desktop)

A desktop application for step-by-step welding inspection using a custom-trained YOLOv8 model.

### Quick Start

```bash
cd yolo
pip install -r requirements.txt
python main.py
```

### Features
- Live camera capture with USB/CSI support
- Real-time YOLOv8 defect detection with bounding boxes
- Automated PDF report generation
- Scan history management
- Raspberry Pi kiosk deployment ready

---

## Raspberry Pi Notes

- App auto-enters **full-screen** on Linux
- Camera: plug in USB webcam → index 0
- CSI camera: `sudo modprobe bcm2835-v4l2` then index 0
- Model path on Pi: set via `WELD_MODEL_PATH` env var or edit `yolo/services/ai_service.py`
