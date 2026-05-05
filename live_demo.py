"""
================================================================================
  ENTERPRISE PPE COMPLIANCE SYSTEM  |  Powered by YOLOv8
  AI-driven real-time safety monitoring for industrial environments.
================================================================================
  Dual-Model Pipeline:
    - yolov8n.pt  →  Person detection  (official COCO model, highly reliable)
    - best.pt     →  Helmet + Vest detection  (custom-trained PPE model)
  Author  : Safety AI Team
  Version : 2.1 (Dual-Model Build)
================================================================================
"""

import cv2
import csv
import time
import sys
import os
from datetime import datetime
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PPE_MODEL_PATH    = "best.pt"       # Custom model: detects helmet + vest
PERSON_MODEL_PATH = "yolov8n.pt"    # Official YOLOv8n: detects persons (COCO)
REPORT_PATH       = "safety_report.csv"
CONFIDENCE        = 0.30            # Detection threshold (0.0 – 1.0)
CAMERA_INDEX      = 0               # 0 = built-in webcam | 1 = external

# HUD layout
HUD_HEIGHT = 100         # Height of the top info bar in pixels
HUD_ALPHA  = 0.75        # Transparency of HUD bar (0 = invisible, 1 = solid)
FONT_MAIN  = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# Color palette (BGR format)
COLOR_SAFE     = (50,  205,  50)   # Lime Green  — all clear
COLOR_CRITICAL = (0,   0,   220)   # Red         — violation detected
COLOR_STANDBY  = (200, 200, 200)   # Light Grey  — no workers in frame
COLOR_ACCENT   = (0,   180, 255)   # Amber       — branding / FPS text
COLOR_WHITE    = (255, 255, 255)
COLOR_DARK     = (20,   20,  20)   # Near-black  — label backgrounds
COLOR_PERSON   = (255, 100,  50)   # Blue-Orange — person bounding box
COLOR_HELMET   = (50,  220,  50)   # Green       — helmet bounding box
COLOR_VEST     = (0,   220, 180)   # Cyan-Teal   — vest bounding box

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(PPE_MODEL_PATH):
    print(f"\n[FATAL ERROR] Model file '{PPE_MODEL_PATH}' not found.")
    print("  Make sure 'best.pt' is in the same folder as live_demo.py")
    sys.exit(1)

try:
    # PPE model — custom trained for helmet and vest detection
    ppe_model   = YOLO(PPE_MODEL_PATH)
    CLASS_NAMES = ppe_model.names       # Expected: {0: 'person', 1: 'helmet', 2: 'vest'}
    print(f"[OK] PPE model loaded     | Classes: {CLASS_NAMES}")

    # Person model — official YOLOv8n (downloads automatically on first run)
    person_model    = YOLO(PERSON_MODEL_PATH)
    PERSON_CLASS_ID = 0                 # COCO class 0 = 'person'
    print(f"[OK] Person model loaded  | yolov8n.pt (COCO)")

except Exception as e:
    print(f"[FATAL ERROR] Could not load model: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  OPEN CAMERA
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[WARN] Camera {CAMERA_INDEX} unavailable. Trying camera 1 ...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[FATAL ERROR] No camera found. Connect a webcam and retry.")
    sys.exit(1)

# Request higher resolution (camera will use closest supported value)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"[OK] Camera opened         | Resolution: "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY REPORT — CSV logging (append mode, writes header only if file is new)
# ─────────────────────────────────────────────────────────────────────────────
csv_file   = open(REPORT_PATH, mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
if os.path.getsize(REPORT_PATH) == 0:
    csv_writer.writerow(['Timestamp', 'Workers', 'Helmets', 'Vests', 'Violations', 'Risk Status'])
print(f"[OK] Report logging active | File: {REPORT_PATH}")
print("\n[SYSTEM READY]  Press Q or ESC to exit.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — draw a labeled stat card on the HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_stat_card(frame, x, y, label, value, color):
    cv2.putText(frame, label,       (x, y),      FONT_SMALL, 0.48, COLOR_STANDBY, 1, cv2.LINE_AA)
    cv2.putText(frame, str(value),  (x, y + 22), FONT_MAIN,  0.85, color,         2, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
violation_flash    = False   # Alternates each frame to create a flashing border
session_violations = 0       # Total CRITICAL incidents recorded this session
fps_timer          = time.time()
fps_value          = 0.0
frame_count        = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Retrying ...")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        frame_count += 1

        # Update FPS counter every 15 frames
        if frame_count % 15 == 0:
            elapsed   = time.time() - fps_timer
            fps_value = 15 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # ── INFERENCE — run both models on the current frame ─────────────
        # Model 1: yolov8n — person detection only (filtered by class ID)
        person_results = person_model(frame, conf=CONFIDENCE, verbose=False,
                                      classes=[PERSON_CLASS_ID])
        # Model 2: best.pt — PPE detection (helmet + vest)
        ppe_results    = ppe_model(frame, conf=CONFIDENCE, verbose=False)

        person_count = 0
        helmet_count = 0
        vest_count   = 0

        # ── DRAW PERSON BOXES (from yolov8n) ────────────────────────────
        for box in person_results[0].boxes:
            conf_score          = float(box.conf[0])
            x1, y1, x2, y2     = map(int, box.xyxy[0])
            person_count       += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2, cv2.LINE_AA)
            label    = f"person  {conf_score:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, FONT_SMALL, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), COLOR_PERSON, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT_SMALL, 0.5, COLOR_DARK, 1, cv2.LINE_AA)

        # ── DRAW HELMET / VEST BOXES (from best.pt) ─────────────────────
        for box in ppe_results[0].boxes:
            cls_id     = int(box.cls[0])
            class_name = CLASS_NAMES.get(cls_id, "unknown")
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == "helmet":
                color = COLOR_HELMET
                helmet_count += 1
            elif class_name == "vest":
                color = COLOR_VEST
                vest_count += 1
            else:
                continue    # Ignore any 'person' detections from best.pt (unreliable)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label    = f"{class_name}  {conf_score:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, FONT_SMALL, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT_SMALL, 0.5, COLOR_DARK, 1, cv2.LINE_AA)

        # ── VIOLATION LOGIC — mathematical deduction ─────────────────────
        missing_helmets  = max(0, person_count - helmet_count)
        missing_vests    = max(0, person_count - vest_count)
        total_violations = missing_helmets + missing_vests

        # ── RISK CLASSIFICATION ──────────────────────────────────────────
        if person_count == 0:
            risk_level  = "STANDBY"
            status_text = "MONITORING ACTIVE  |  NO WORKERS DETECTED"
            hud_color   = COLOR_STANDBY

        elif total_violations == 0:
            risk_level  = "SAFE"
            status_text = "STATUS: SAFE  |  FULL PPE COMPLIANCE"
            hud_color   = COLOR_SAFE

        else:
            risk_level  = "CRITICAL"
            status_text = f"ALERT: {total_violations} VIOLATION(S) DETECTED  —  TAKE ACTION"
            hud_color   = COLOR_CRITICAL
            session_violations += 1

            # Log the incident to CSV immediately
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, person_count, helmet_count,
                                 vest_count, total_violations, "CRITICAL"])
            csv_file.flush()

            # Flash a red border around the frame as a visual alarm
            violation_flash = not violation_flash
            if violation_flash:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_CRITICAL, 12)

        # Reset flash when status is no longer critical
        if risk_level != "CRITICAL":
            violation_flash = False

        # ── HUD — semi-transparent top bar ───────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, HUD_HEIGHT), COLOR_DARK, -1)
        frame   = cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0)

        # Left accent bar (color reflects current risk state)
        cv2.rectangle(frame, (0, 0), (6, HUD_HEIGHT), hud_color, -1)

        # System branding — top left
        cv2.putText(frame, "PPE COMPLIANCE SYSTEM",
                    (16, 28), FONT_MAIN, 0.65, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Powered by YOLOv8  |  Real-Time AI",
                    (16, 50), FONT_SMALL, 0.4, COLOR_STANDBY, 1, cv2.LINE_AA)

        # Main status message
        cv2.putText(frame, status_text,
                    (16, 82), FONT_MAIN, 0.62, hud_color, 2, cv2.LINE_AA)

        # Stat cards — right side of HUD
        card_y   = 28
        card_gap = 115
        right    = w - 480
        draw_stat_card(frame, right,              card_y, "WORKERS",    person_count,      COLOR_PERSON)
        draw_stat_card(frame, right + card_gap,   card_y, "HELMETS",    helmet_count,      COLOR_HELMET)
        draw_stat_card(frame, right + card_gap*2, card_y, "VESTS",      vest_count,        COLOR_VEST)
        draw_stat_card(frame, right + card_gap*3, card_y, "VIOLATIONS", total_violations,
                       COLOR_CRITICAL if total_violations > 0 else COLOR_SAFE)

        # Metadata — bottom corners
        cv2.putText(frame, f"Session Incidents: {session_violations}",
                    (10, h - 12), FONT_SMALL, 0.42, COLOR_STANDBY, 1, cv2.LINE_AA)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    (w - 260, h - 12), FONT_SMALL, 0.42, COLOR_STANDBY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps_value:.1f}",
                    (w - 90, h - 12), FONT_SMALL, 0.42, COLOR_ACCENT, 1, cv2.LINE_AA)

        # ── DISPLAY ───────────────────────────────────────────────────────
        cv2.imshow("Enterprise PPE Compliance System  |  Press Q to Exit", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):  # Q or ESC
            break

# ─────────────────────────────────────────────────────────────────────────────
#  GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"\n[ERROR] Unexpected crash: {e}")
finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[SESSION SUMMARY]")
    print(f"  Total incidents logged : {session_violations}")
    print(f"  Report saved to        : {REPORT_PATH}")
    print("  System shut down cleanly.\n")