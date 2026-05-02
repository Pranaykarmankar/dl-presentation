"""
services/report_service.py
──────────────────────────
Thin wrapper around welding_report.generate_report().

The controller calls generate() and gets back a PDF path.
No UI code lives here — only data transformation and file I/O.

When the real AI model is integrated, DetectionResult boxes from
ai_service.py should be converted here into the defects_found list
format that welding_report.py expects.
"""

import os
import datetime
from utils.constants import BASE_DIR

# Ensure welding_report.py is importable from project root
import sys
sys.path.insert(0, BASE_DIR)

try:
    from welding_report import generate_report as _generate_report
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


class ReportService:
    """
    Generates a PDF inspection report from scan metadata and AI results.

    Usage:
        svc = ReportService()
        path = svc.generate(scan_dict, ai_result_or_None)
    """

    @property
    def available(self) -> bool:
        return REPORTLAB_AVAILABLE

    def generate(self, scan: dict, ai_result=None) -> str:
        """
        Build and save a PDF report.

        Args:
            scan       : scan metadata dict (from scans.json)
            ai_result  : DetectionResult from AIService, or None (placeholder mode)

        Returns:
            Absolute path to the saved PDF file.
        """
        report_id = scan.get('report_id', 'RPT-UNKNOWN')
        ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f"{report_id}_{ts}.pdf"
        out_path  = os.path.join(REPORTS_DIR, filename)

        scan_data    = self._build_scan_data(scan)
        defects_found = self._build_defects(ai_result)

        if REPORTLAB_AVAILABLE:
            _generate_report(scan_data, defects_found, out_path)
        else:
            # Write a plain-text stub so the viewer has something to open
            with open(out_path.replace('.pdf', '.txt'), 'w') as f:
                f.write(f"Report: {report_id}\n")
                f.write("ReportLab not installed — install with: pip install reportlab\n")
            out_path = out_path.replace('.pdf', '.txt')

        return out_path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_scan_data(self, scan: dict) -> dict:
        """Map the app's scan dict to the format welding_report.py expects."""
        w = scan.get('width', '?')
        h = scan.get('height', '?')
        return {
            'report_id':      scan.get('report_id', '—'),
            'sample_id':      scan.get('sample_id', '—'),
            'scan_number':    scan.get('scan_number', '—'),
            'scan_date':      scan.get('scan_date', '—'),
            'operator_name':  scan.get('operator', '—'),
            'location':       scan.get('location', '—'),
            'specimen_type':  scan.get('specimen_type', '—'),
            'joint_config':   scan.get('joint_config', '—'),
            'material':       scan.get('material', '—'),
            'specimen_size':  f"{w} x {h} mm",
            'welding_process': scan.get('welding_process', '—'),
            'industry':       scan.get('industry', 'General'),
            'standard':       'ISO 5817 Level B / API 1104',
            'model_version':  'v1 (Simulation)',
            'device':         'Inspection Camera',
            'company_name':   'DSES Weld Inspector',
            'scan_side':      'CAP',
            'confidence_avg': scan.get('confidence_avg', 0.0),
            'overall_verdict': scan.get('overall_verdict', 'PENDING ANALYSIS'),
        }

    def _build_defects(self, ai_result) -> list:
        """
        Convert an AIService DetectionResult into the defects_found list.

        When ai_result is None (placeholder / pre-ML), returns an empty list
        so the report correctly shows "No defects detected."

        ── TO INTEGRATE THE REAL MODEL ──────────────────────────────────────
        When AIService.run_inference() returns a real DetectionResult, the
        boxes list will have BoundingBox objects with .label and .confidence.
        Group them by label here and fill in image_path from the captured photo.
        ─────────────────────────────────────────────────────────────────────
        """
        if ai_result is None:
            return []

        # Group boxes by defect label
        groups: dict[str, list] = {}
        for box in getattr(ai_result, 'boxes', []):
            groups.setdefault(box.label, []).append(box)

        defects = []
        for label, boxes in groups.items():
            avg_conf = sum(b.confidence for b in boxes) / len(boxes) * 100
            defects.append({
                'type':           _normalise_label(label),
                'count':          len(boxes),
                'avg_confidence': round(avg_conf, 1),
                'locations':      'Detected by AI model',
                'image_path':     None,
            })
        return defects


def _normalise_label(raw: str) -> str:
    """Map model output labels to DEFECT_DATABASE keys."""
    mapping = {
        'porosity':           'Porosity',
        'crack':              'Crack',
        'undercut':           'Undercut',
        'overlap':            'Overlap',
        'incomplete_fusion':  'Lack of Fusion',
        'spatter':            'Spatter',
        'slag_inclusion':     'Slag Inclusion',
        'burn_through':       'Burn Through',
        'lack_of_penetration':'Lack of Penetration',
        'underfill':          'Underfill',
        'arc_strike':         'Arc Strike',
        'mechanical_mark':    'Mechanical Mark',
    }
    return mapping.get(raw.lower(), raw.replace('_', ' ').title())
