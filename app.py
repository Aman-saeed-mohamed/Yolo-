import cv2
import csv
import time
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify
import threading

app = Flask(__name__)

try:
    model = YOLO("best.pt")
except Exception as e:
    print("Error: Make sure 'best.pt' is in the exact same folder as this script!")
    exit()

CLASS_NAMES = model.names

csv_file = open('safety_report.csv', mode='a', newline='')
writer = csv.writer(csv_file)

current_stats = {
    "workers": 0,
    "helmets": 0,
    "vests": 0,
    "violations": 0,
    "status": "STANDBY"
}

latest_frame = None
lock = threading.Lock()

def process_video():
    global current_stats, latest_frame
    
    # محاولة فتح الكاميرا (نجرب 0 ثم 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # استخدام الدقة الافتراضية وتقليل نسبة الثقة (conf) لتحسين فرصة الاكتشاف
        results = model(frame, conf=0.25, verbose=False)
        
        person_count = 0
        helmet_count = 0
        vest_count = 0
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(cls_id, "unknown")
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if class_name == "person":
                color = (255, 0, 0)
                person_count += 1
            elif class_name == "helmet":
                color = (0, 255, 0)
                helmet_count += 1
            elif class_name == "vest":
                color = (0, 255, 255)
                vest_count += 1
            else:
                color = (255, 255, 255)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        missing_helmets = max(0, person_count - helmet_count)
        missing_vests = max(0, person_count - vest_count)
        total_violations = missing_helmets + missing_vests
        
        if person_count == 0:
            status = "STANDBY"
        elif total_violations == 0:
            status = "SAFE"
        else:
            status = "CRITICAL"
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([current_time, person_count, helmet_count, vest_count, total_violations, "CRITICAL"])
            csv_file.flush()
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        with lock:
            current_stats["workers"] = person_count
            current_stats["helmets"] = helmet_count
            current_stats["vests"] = vest_count
            current_stats["violations"] = total_violations
            current_stats["status"] = status
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                latest_frame = buffer.tobytes()

threading.Thread(target=process_video, daemon=True).start()

def generate():
    while True:
        with lock:
            frame = latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with lock:
        return jsonify(current_stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
