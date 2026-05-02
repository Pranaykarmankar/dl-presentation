"""
services/ai_service.py
──────────────────────
AI / ML inference service for weld defect detection.

Current status: PLACEHOLDER — returns simulated results.

To integrate RTMDet (or any ONNX model):
  1. Place model file at:  <project_root>/model/end2end.onnx
  2. Set MODEL_PATH below (or pass it to AIService.__init__)
  3. Implement _load_model() and run_inference() using onnxruntime.

The rest of the application (controller, UI) calls only run_inference()
and reads DetectionResult objects — no other changes needed.
"""

import os
import sys
import random
import datetime
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Model path ───────────────────────────────────────────────────────────────
# On Raspberry Pi the model lives at:
#   /home/pi/Desktop/Weld-Inspection/model/end2end.onnx
# On other systems set the env var  WELD_MODEL_PATH  or change this default.
_DEFAULT_MODEL = os.environ.get(
    '/Users/raunak/Downloads/weld_inspector/model/best.onnx',
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'end2end.onnx')
)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str          # e.g. "porosity", "crack", "undercut"
    label_id: int = 0


@dataclass
class DetectionResult:
    boxes: List[BoundingBox]   = field(default_factory=list)
    inference_time_ms: float   = 0.0
    frame_width: int           = 0
    frame_height: int          = 0
    is_simulated: bool         = True   # False once real model is loaded
    timestamp: str             = ''

    @property
    def defect_count(self) -> int:
        return len(self.boxes)

    @property
    def has_defects(self) -> bool:
        return bool(self.boxes)

    @property
    def severity(self) -> str:
        n = self.defect_count
        if n == 0:  return 'Pass'
        if n <= 2:  return 'Minor'
        if n <= 5:  return 'Moderate'
        return 'Severe'


# ─── Service ──────────────────────────────────────────────────────────────────

class AIService:
    """
    Weld defect detection service.

    Loads an ONNX model on Linux/RPi if onnxruntime is installed.
    Falls back to a deterministic simulation on other platforms.
    """

    # Input resolution expected by the model
    INPUT_W = 640
    INPUT_H = 640

    # Confidence threshold — detections below this are discarded
    CONF_THRESHOLD = 0.45

    # Class names produced by the model (index → label)
    CLASS_NAMES = [
        'porosity', 'crack', 'undercut', 'overlap',
        'incomplete_fusion', 'spatter', 'slag_inclusion',
    ]

    def __init__(self, model_path: str = _DEFAULT_MODEL):
        self._model_path = model_path
        self._session = None
        self._load_model()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_model(self):
        """Load ONNX model if available; silently fall back to simulation."""
        if not _ORT_AVAILABLE:
            return
        if not os.path.exists(self._model_path):
            return
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._session = ort.InferenceSession(self._model_path, providers=providers)
        except Exception as e:
            print(f'[AIService] Could not load model: {e}')
            self._session = None

    @property
    def model_loaded(self) -> bool:
        return self._session is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def run_inference(self, frame) -> DetectionResult:
        """
        Run defect detection on a BGR OpenCV frame.
        Returns a DetectionResult.  Always succeeds (falls back to simulation).
        """
        ts = datetime.datetime.now().isoformat(timespec='milliseconds')
        if frame is None:
            return DetectionResult(timestamp=ts)

        h, w = frame.shape[:2]

        if self.model_loaded:
            return self._real_inference(frame, w, h, ts)
        else:
            return self._simulated_inference(w, h, ts)

    # ── Real inference (ONNX) ─────────────────────────────────────────────────

    def _real_inference(self, frame, w: int, h: int, ts: str) -> DetectionResult:
        """
        Preprocess → infer → postprocess using the loaded ONNX session.

        ┌─ TODO when you have the real model ─────────────────────────────────┐
        │  Adjust the input tensor name (self._session.get_inputs()[0].name)  │
        │  and the output parsing to match your model's output format.        │
        └─────────────────────────────────────────────────────────────────────┘
        """
        try:
            import cv2
            import numpy as np
            import time

            t0     = time.perf_counter()
            resized = cv2.resize(frame, (self.INPUT_W, self.INPUT_H))
            blob    = resized.astype('float32') / 255.0
            blob    = blob.transpose(2, 0, 1)[None]          # NCHW

            input_name = self._session.get_inputs()[0].name
            outputs    = self._session.run(None, {input_name: blob})

            # ── Parse output ─────────────────────────────────────────────────
            # Adjust the slice indices to match your model's output format.
            # Typical end2end format: [batch, num_dets, 6]  (x1,y1,x2,y2,conf,cls)
            dets   = outputs[0][0]            # shape: (N, 6)
            elapsed = (time.perf_counter() - t0) * 1000

            boxes = []
            scale_x = w / self.INPUT_W
            scale_y = h / self.INPUT_H
            for det in dets:
                x1, y1, x2, y2, conf, cls_id = det
                if conf < self.CONF_THRESHOLD:
                    continue
                label = self.CLASS_NAMES[int(cls_id)] if int(cls_id) < len(self.CLASS_NAMES) else 'defect'
                boxes.append(BoundingBox(
                    x1=float(x1) * scale_x, y1=float(y1) * scale_y,
                    x2=float(x2) * scale_x, y2=float(y2) * scale_y,
                    confidence=float(conf), label=label, label_id=int(cls_id),
                ))

            return DetectionResult(
                boxes=boxes, inference_time_ms=elapsed,
                frame_width=w, frame_height=h,
                is_simulated=False, timestamp=ts,
            )

        except Exception as e:
            print(f'[AIService] Inference error: {e}')
            return self._simulated_inference(w, h, ts)

    # ── Simulation (non-RPi / no model) ──────────────────────────────────────

    def _simulated_inference(self, w: int, h: int, ts: str) -> DetectionResult:
        """Return plausible-looking fake results for UI development."""
        rng   = random.Random(hash(ts) % 2**32)
        count = rng.randint(0, 3)
        boxes = []
        for _ in range(count):
            x1 = rng.uniform(0.15, 0.55) * w
            y1 = rng.uniform(0.15, 0.55) * h
            x2 = x1 + rng.uniform(30, 100)
            y2 = y1 + rng.uniform(20, 60)
            label = rng.choice(self.CLASS_NAMES[:4])
            boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=min(x2, w), y2=min(y2, h),
                confidence=rng.uniform(0.55, 0.97),
                label=label,
            ))
        return DetectionResult(
            boxes=boxes, inference_time_ms=rng.uniform(18, 45),
            frame_width=w, frame_height=h,
            is_simulated=True, timestamp=ts,
        )
