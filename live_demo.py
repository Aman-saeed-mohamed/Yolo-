"""
================================================================================
  ENTERPRISE PPE COMPLIANCE SYSTEM  |  Powered by YOLOv8
  AI-driven real-time safety monitoring for industrial environments.
================================================================================
  Pipeline:
    - yolov8n.pt     →  Person detection  (official COCO model)
    - best_new.pt    →  Helmet detection  (keremberke/yolov8n-hard-hat-detection)
    - HSV color      →  Vest detection    (Hi-Vis yellow/orange color analysis)
  Version : 3.0 (Clean Build)
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
HELMET_MODEL_PATH = "best_new.pt"   # keremberke/yolov8n-hard-hat-detection
PERSON_MODEL_PATH = "yolov8n.pt"    # COCO person detector
REPORT_PATH       = "safety_report.csv"
CONFIDENCE        = 0.30            # YOLO confidence threshold
CAMERA_INDEX      = 1               # 0 = built-in webcam | 1 = DroidCam/external
VEST_RATIO_MIN    = 0.06            # Min vest-color ratio in torso to count as vest
VEST_PIXEL_MIN    = 400             # Min absolute vest-colored pixels (noise filter)

# HUD layout
HUD_HEIGHT = 100
HUD_ALPHA  = 0.75
FONT_MAIN  = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# Color palette (BGR)
COLOR_SAFE     = (50,  205,  50)
COLOR_CRITICAL = (0,   0,   220)
COLOR_STANDBY  = (200, 200, 200)
COLOR_ACCENT   = (0,   180, 255)
COLOR_WHITE    = (255, 255, 255)
COLOR_DARK     = (20,   20,  20)
COLOR_PERSON   = (255, 100,  50)
COLOR_HELMET   = (50,  220,  50)
COLOR_VEST     = (0,   220, 180)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
for path in (HELMET_MODEL_PATH, PERSON_MODEL_PATH):
    if not os.path.exists(path):
        print(f"\n[FATAL ERROR] Model file '{path}' not found.")
        sys.exit(1)

try:
    helmet_model    = YOLO(HELMET_MODEL_PATH)
    HELMET_CLASSES  = helmet_model.names    # {0:'Hardhat', 1:'NO-Hardhat'}
    print(f"[OK] Helmet model  | {HELMET_CLASSES}")

    person_model    = YOLO(PERSON_MODEL_PATH)
    PERSON_CLASS_ID = 0
    print(f"[OK] Person model  | yolov8n (COCO)")

except Exception as e:
    print(f"[FATAL ERROR] {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  OPEN CAMERA
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[WARN] Camera {CAMERA_INDEX} unavailable. Trying camera 0 ...")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[FATAL ERROR] No camera found.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"[OK] Camera  | {int(cap.get(3))}x{int(cap.get(4))}")

# Drain warm-up frames (DroidCam / USB cameras need a moment)
for _ in range(25):
    cap.read()

# ─────────────────────────────────────────────────────────────────────────────
#  CSV REPORT
# ─────────────────────────────────────────────────────────────────────────────
csv_file   = open(REPORT_PATH, mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
if os.path.getsize(REPORT_PATH) == 0:
    csv_writer.writerow(['Timestamp', 'Workers', 'Helmets', 'Vests', 'Violations', 'Risk'])
print(f"[OK] Report    | {REPORT_PATH}")
print("\n[SYSTEM READY]  Press Q or ESC to exit.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  VEST DETECTION — Morphological kernel (built once, reused every frame)
# ─────────────────────────────────────────────────────────────────────────────
VEST_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_NAME = "Enterprise PPE Compliance System  |  Press Q to Exit"

def draw_stat_card(frame, x, y, label, value, color):
    cv2.putText(frame, label,      (x, y),      FONT_SMALL, 0.48, COLOR_STANDBY, 1, cv2.LINE_AA)
    cv2.putText(frame, str(value), (x, y + 22), FONT_MAIN,  0.85, color,         2, cv2.LINE_AA)

def draw_label(frame, text, x1, y1, bg_color, text_color=None):
    """Draw a filled label rectangle above (x1, y1)."""
    if text_color is None:
        text_color = COLOR_DARK
    (lw, lh), _ = cv2.getTextSize(text, FONT_SMALL, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), bg_color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), FONT_SMALL, 0.5, text_color, 1, cv2.LINE_AA)

def overlaps_with_person(hx1, hy1, hx2, hy2, person_boxes,
                         min_iou=0.10, zone='full'):
    """
    Returns the index of the matching person, or -1 if none.

    zone='head'  → additionally checks that the helmet center is in the
                   top 30% of the person box (head region).
    zone='torso' → center must be in the 25%-75% band.
    zone='full'  → just overlap check, no zone restriction.
    """
    hcx = (hx1 + hx2) // 2   # helmet center X
    hcy = (hy1 + hy2) // 2   # helmet center Y
    h_area = max(1, (hx2 - hx1) * (hy2 - hy1))

    for idx, (px1, py1, px2, py2) in enumerate(person_boxes):
        # ── overlap check ────────────────────────────────────────────────
        ix1 = max(hx1, px1);  iy1 = max(hy1, py1)
        ix2 = min(hx2, px2);  iy2 = min(hy2, py2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        if inter / h_area < min_iou:
            continue

        # ── zone check ───────────────────────────────────────────────────
        ph = max(1, py2 - py1)
        rel_y = (hcy - py1) / ph   # 0.0 = top of person, 1.0 = bottom

        if zone == 'head'  and rel_y > 0.30:
            continue   # helmet center is NOT in head zone → skip
        if zone == 'torso' and not (0.25 <= rel_y <= 0.75):
            continue   # not in torso zone → skip

        return idx
    return -1

def detect_vest_hsv(frame, person_boxes):
    """
    Returns a set of person indices wearing a Hi-Vis vest.
    Checks only the inner torso zone to avoid background interference.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Hi-Vis Yellow-Green: high saturation + high brightness (fluorescent)
    # S >= 120, V >= 120 to exclude dull/dark backgrounds
    m1 = cv2.inRange(hsv, (22, 120, 120), (95, 255, 255))   # Yellow / Lime / Green
    # Orange: also fluorescent safety vests
    m2 = cv2.inRange(hsv, (5,  120, 120), (22, 255, 255))   # Orange

    # NOTE: Red range removed — causes false positives with red curtains/walls.
    # Safety vests are almost never pure red; they are yellow-green or orange.

    vest_mask = cv2.bitwise_or(m1, m2)
    vest_mask = cv2.morphologyEx(vest_mask, cv2.MORPH_OPEN, VEST_KERNEL)

    wearing = set()
    fh, fw  = frame.shape[:2]

    for idx, (px1, py1, px2, py2) in enumerate(person_boxes):
        ph = py2 - py1
        pw = px2 - px1
        if ph < 80 or pw < 50:
            continue

        # Strict torso zone: 30% → 70% of person height
        tz1 = max(0,  py1 + int(ph * 0.30))
        tz2 = min(fh, py1 + int(ph * 0.70))

        # Horizontal margin: use inner 70% of person width to avoid background
        margin = int(pw * 0.15)
        cx1 = max(0,  px1 + margin)
        cx2 = min(fw, px2 - margin)

        if cx2 <= cx1:
            continue

        crop = vest_mask[tz1:tz2, cx1:cx2]
        if crop.size == 0:
            continue

        vp    = cv2.countNonZero(crop)
        ratio = vp / crop.size

        # Requires >15% of inner torso is vest-colored AND at least 1000 pixels
        if ratio > 0.15 and vp > 1000:
            wearing.add(idx)

    return wearing, vest_mask


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
violation_flash    = False
session_violations = 0
fps_timer          = time.time()
fps_value          = 0.0
frame_count        = 0

# Create fullscreen window BEFORE the loop (must be before first imshow on Windows)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        frame_count += 1

        if frame_count % 15 == 0:
            elapsed   = time.time() - fps_timer
            fps_value = 15 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # ── INFERENCE ────────────────────────────────────────────────────
        person_results = person_model(frame, conf=CONFIDENCE, verbose=False,
                                      classes=[PERSON_CLASS_ID])
        helmet_results = helmet_model(frame, conf=CONFIDENCE, verbose=False)

        person_count = 0
        helmet_count = 0
        vest_count   = 0
        person_boxes = []

        # ── PERSONS ──────────────────────────────────────────────────────
        for box in person_results[0].boxes:
            conf_score      = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_count   += 1
            person_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2, cv2.LINE_AA)
            draw_label(frame, f"person  {conf_score:.0%}", x1, y1, COLOR_PERSON)

        # ── HELMETS — only count if on a detected person ─────────────────
        persons_with_helmet = set()

        for box in helmet_results[0].boxes:
            cls_id     = int(box.cls[0])
            class_name = HELMET_CLASSES.get(cls_id, "unknown")
            conf_score = float(box.conf[0])
            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])

            # Helmet must overlap with a person AND be in the HEAD zone (top 30%)
            person_idx = overlaps_with_person(hx1, hy1, hx2, hy2, person_boxes,
                                              min_iou=0.10, zone='head')

            if person_idx == -1:
                continue   # helmet/NO-Hardhat not on any person — ignore

            if class_name == "Hardhat":
                helmet_count += 1
                persons_with_helmet.add(person_idx)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), COLOR_HELMET, 2, cv2.LINE_AA)
                draw_label(frame, f"Helmet  {conf_score:.0%}", hx1, hy1, COLOR_HELMET)

            elif class_name == "NO-Hardhat":
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), COLOR_CRITICAL, 3, cv2.LINE_AA)
                draw_label(frame, f"NO HELMET  {conf_score:.0%}", hx1, hy1,
                           COLOR_CRITICAL, COLOR_WHITE)

        # ── VEST DETECTION (HSV) ─────────────────────────────────────────
        wearing_vest, _ = detect_vest_hsv(frame, person_boxes)
        vest_count = len(wearing_vest)

        for idx, (px1, py1, px2, py2) in enumerate(person_boxes):
            fh = frame.shape[0]
            tz1 = max(0, py1 + (py2 - py1) // 4)
            tz2 = min(fh, py2)

            if idx in wearing_vest:
                # Teal box around torso
                cv2.rectangle(frame, (px1, tz1), (px2, tz2), COLOR_VEST, 2, cv2.LINE_AA)
                draw_label(frame, "Vest ✓", px1, tz1, COLOR_VEST)
            else:
                # Red NO VEST label at mid-person height
                my = (py1 + py2) // 2
                draw_label(frame, "NO VEST", px1, my + 12, COLOR_CRITICAL, COLOR_WHITE)

        # ── VIOLATIONS ───────────────────────────────────────────────────
        missing_helmets  = max(0, person_count - helmet_count)
        missing_vests    = max(0, person_count - vest_count)
        total_violations = missing_helmets + missing_vests

        # ── RISK LEVEL ───────────────────────────────────────────────────
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

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, person_count, helmet_count,
                                 vest_count, total_violations, "CRITICAL"])
            csv_file.flush()

            violation_flash = not violation_flash
            if violation_flash:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_CRITICAL, 12)

        if risk_level != "CRITICAL":
            violation_flash = False

        # ── HUD ──────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, HUD_HEIGHT), COLOR_DARK, -1)
        frame = cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0)
        cv2.rectangle(frame, (0, 0), (6, HUD_HEIGHT), hud_color, -1)

        cv2.putText(frame, "PPE COMPLIANCE SYSTEM",
                    (16, 28), FONT_MAIN, 0.65, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Powered by YOLOv8  |  Real-Time AI",
                    (16, 50), FONT_SMALL, 0.4, COLOR_STANDBY, 1, cv2.LINE_AA)
        cv2.putText(frame, status_text,
                    (16, 82), FONT_MAIN, 0.62, hud_color, 2, cv2.LINE_AA)

        card_y   = 28
        card_gap = 115
        right    = w - 480
        draw_stat_card(frame, right,              card_y, "WORKERS",    person_count, COLOR_PERSON)
        draw_stat_card(frame, right + card_gap,   card_y, "HELMETS",    helmet_count, COLOR_HELMET)
        draw_stat_card(frame, right + card_gap*2, card_y, "VESTS",      vest_count,   COLOR_VEST)
        draw_stat_card(frame, right + card_gap*3, card_y, "VIOLATIONS", total_violations,
                       COLOR_CRITICAL if total_violations > 0 else COLOR_SAFE)

        cv2.putText(frame, f"Session Incidents: {session_violations}",
                    (10, h - 12), FONT_SMALL, 0.42, COLOR_STANDBY, 1, cv2.LINE_AA)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    (w - 260, h - 12), FONT_SMALL, 0.42, COLOR_STANDBY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps_value:.1f}",
                    (w - 90, h - 12), FONT_SMALL, 0.42, COLOR_ACCENT, 1, cv2.LINE_AA)

        # ── DISPLAY ──────────────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

# ─────────────────────────────────────────────────────────────────────────────
#  SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
except KeyboardInterrupt:
    print("\n[INFO] Interrupted.")
except Exception as e:
    print(f"\n[ERROR] {e}")
finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[SESSION] Incidents: {session_violations} | Report: {REPORT_PATH}\n")