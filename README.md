# 🦺 Enterprise Real-Time PPE Compliance & Risk Monitoring System
### Powered by YOLOv8 · Built with OpenCV · Eastern Mediterranean University

> An AI-driven safety monitoring system that autonomously detects workers and verifies PPE compliance (helmets & vests) in real time — replacing manual safety auditing with proactive, automated risk prevention.

---

## 🎯 Project Overview

Workplace accidents in construction and industrial environments result in severe injuries and massive financial losses. This system introduces an **enterprise-grade, real-time AI monitoring solution** that:

- 📷 Captures a **live camera feed** directly on a local machine (no cloud, no internet required)
- 🧠 Runs **YOLOv8 inference** to detect workers and their safety equipment every frame
- ⚠️ Uses a **Mathematical Logic Layer** to deduce PPE violations without needing a "no-helmet" class
- 📊 Displays a live **Heads-Up Display (HUD)** with risk status, worker counts, and alerts
- 📝 **Auto-logs all violations** to a timestamped CSV file for management reporting

---

## 🧠 How It Works — The Logic Layer

The model detects **3 core classes only**:

| ID | Class | Color on Screen |
|----|-------|----------------|
| 0  | `person` | 🔵 Blue-Orange |
| 1  | `helmet` | 🟢 Green |
| 2  | `vest`   | 🩵 Cyan-Teal |

Violations are deduced mathematically — no extra training needed:

```
missing_helmets  = max(0, person_count - helmet_count)
missing_vests    = max(0, person_count - vest_count)
total_violations = missing_helmets + missing_vests
```

This optimization makes the system **faster** and proves **deep engineering problem-solving**.

---

## 🖥️ Risk States

| State | Trigger | HUD Color | Effect |
|-------|---------|-----------|--------|
| **STANDBY** | No workers detected | ⚪ Grey | Monitoring active message |
| **SAFE** | All workers have PPE | 🟢 Green | Full compliance message |
| **CRITICAL** | Any PPE missing | 🔴 Red | Flashing red border + violation count + CSV log |

---

## 📁 Project Structure

```
Yolo/
├── live_demo.py          # 🎬 Main stage demo — pure OpenCV, crash-proof
├── app.py                # 🌐 Flask web dashboard (optional)
├── templates/
│   ├── index.html        # Landing page
│   └── monitor.html      # Live monitoring dashboard
├── static/               # CSS / JS assets
├── best.pt               # 🧠 Trained YOLOv8 model weights
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Aman-saeed-mohamed/Yolo-.git
cd Yolo-
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Live Demo

```bash
python live_demo.py
```

- Press **`Q`** or **`ESC`** to exit cleanly.
- Violations are automatically saved to `safety_report.csv`.

---

## 📊 Automated Report (safety_report.csv)

Every CRITICAL event is instantly logged:

| Timestamp | Workers | Helmets | Vests | Violations | Risk Status |
|-----------|---------|---------|-------|------------|-------------|
| 2026-05-05 10:24:36 | 3 | 1 | 2 | 2 | CRITICAL |

---

## 🛠️ Requirements

```
ultralytics
opencv-python
flask
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 💡 Business Value

| Benefit | Impact |
|---------|--------|
| ✅ Replaces manual safety inspectors | Reduces 24/7 labor costs |
| ✅ Instant violation detection | Prevents accidents before they happen |
| ✅ Automated compliance reports | Eliminates manual paperwork |
| ✅ Runs 100% locally | Zero cloud costs, full data privacy |
| ✅ Works on standard laptops | No specialized hardware needed |

---

## 👤 Authors

**Aman Saeed Mohamed** — AI Developer & Data Analyst
**Mustapha Ali Gumel** — Team Member
**Mentor: Akile ODAY**

🔗 [GitHub](https://github.com/Aman-saeed-mohamed) · [LinkedIn](https://www.linkedin.com/in/aman-saeed-mo/)

---

## 📚 References

- Jocher, G., et al. (2023). *Ultralytics YOLOv8*. [github.com/ultralytics](https://github.com/ultralytics/ultralytics)
- Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal.
- OSHA (2022). *Personal Protective Equipment Guidelines*. [osha.gov](https://www.osha.gov/personal-protective-equipment)

---

*📌 This project is developed for academic and research purposes at Eastern Mediterranean University.*