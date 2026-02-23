# Blinky: Computer Vision Eye-Health Utility

**Blinky** is a Python-based utility designed to mitigate Digital Eye Strain (DES) and monitor facial tension (scowling) during prolonged computer use. It uses real-time computer vision to track user behavior and provides hardware-level feedback to encourage healthier habits.

---

## Core Functionality

The script utilizes a real-time inference pipeline to monitor two primary metrics:

* **Blink Rate (BPM):** Measures Blinks-Per-Minute using the **Eye Aspect Ratio (EAR)**. If the rate falls below the threshold (15 BPM), the system triggers a "dimmed" state.

* **Glabella Tension (Scowling):** Monitors the distance between inner brow landmarks and the nasal bridge.
    * **Continuous Scowl:** If a frown is held for >5 seconds, the screen dims.
    * **Cumulative Scowl:** If total scowl time exceeds a set limit (e.g., 15s) within a rolling 60-second window, an audio alert is triggered.

---

## The Biofeedback Loop

To restore screen brightness from the dimmed state (**20%**), the user must complete a **Deep-Blink sequence**. This requires five deliberate eye closures, each held for at least 0.5 seconds. This forced movement ensures the ocular surface is properly lubricated and the facial muscles are relaxed before resuming work.

---

## Technical Stack

* **Inference:** MediaPipe (Face Mesh 468 Landmarks)
* **Vision:** OpenCV (cv2)
* **System Control:** Screen-Brightness-Control API
* **Math:** Scipy (Euclidean distance for EAR and Frown Scoring)

---

## Installation & Usage

### 1. Clone the repository:
```bash
git clone [https://github.com/0ut1ander/Blinky-The-AI-Screen-Strain-Solution.git](https://github.com/0ut1ander/Blinky-The-AI-Screen-Strain-Solution.git)
cd Blinky-The-AI-Screen-Strain-Solution