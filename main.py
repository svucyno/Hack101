# main.py

import torch
import sys
import os
from collections import deque
from ultralytics import YOLO
import cv2
from utils import decision_and_alert
from lstm import LSTMAccidentDetector

# ─── Model Load ───────────────────────────────────────────────
trained_path = 'runs/detect/accident_cnn/weights/best.pt'
model = YOLO(trained_path if os.path.exists(trained_path) else "yolov8n.pt")
print(f"[MODEL] {'Trained model' if os.path.exists(trained_path) else 'Base YOLOv8'} loaded.")

# ─── Output Folder ────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

# ─── Video Path ───────────────────────────────────────────────
# Run: python main.py "path/to/video.mp4"
# Or drop your video path below as default
DEFAULT_VIDEO = "input.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
print(f"[VIDEO] Processing: {video_path}")

# ─── LSTM Setup ───────────────────────────────────────────────
lstm_model = LSTMAccidentDetector(input_size=6).eval()
sequence = deque(maxlen=10)

# ─── Open Video ───────────────────────────────────────────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Video open cheyaledu: {video_path}")
    sys.exit(1)

frame_count = 0
print("[START] Detection starting...")

# ─── Main Loop ────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("[DONE] Video processing complete.")
        break

    frame_count += 1

    # YOLOv8 detection
    results = model(frame, verbose=False)

    acc_conf = fire_conf = norm_conf = 0.0
    acc_count = fire_count = norm_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id == 0:      # accident
                acc_conf  = max(acc_conf, conf)
                acc_count += 1
            elif cls_id == 1:    # fire
                fire_conf  = max(fire_conf, conf)
                fire_count += 1
            elif cls_id == 2:    # normal
                norm_conf  = max(norm_conf, conf)
                norm_count += 1

    # LSTM sequence build
    features = [
        acc_conf,  fire_conf,  norm_conf,
        float(acc_count)  / 10.0,
        float(fire_count) / 10.0,
        float(norm_count) / 10.0
    ]
    sequence.append(features)

    # ─── Decision Logic ───────────────────────────────────────
    is_accident = False
    is_fire     = False

    if len(sequence) == 10:
        # LSTM tho confirm cheyali
        seq_tensor = torch.tensor(list(sequence)).unsqueeze(0).float()
        probs = lstm_model(seq_tensor)
        if probs[0, 0] > 0.5:
            is_accident = True
        if probs[0, 1] > 0.5:
            is_fire = True
    else:
        # LSTM ready kaakamunde — direct confidence threshold
        if acc_conf  > 0.3:
            is_accident = True
        if fire_conf > 0.3:
            is_fire = True

    # ─── Alert + Proof Image ──────────────────────────────────
    if is_accident or is_fire:
        # Proof image save
        img_path = f"output/frame_{frame_count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"[DETECTED] Frame {frame_count} | Accident: {is_accident} | Fire: {is_fire}")
        print(f"[PROOF] Saved: {img_path}")

        # Alert pampu
        decision_and_alert(is_accident, is_fire, img_path)

    # ─── Live Display ─────────────────────────────────────────
    # Detection boxes draw cheyali
    annotated = results[0].plot()
    label = ""
    if is_fire:
        label = "FIRE DETECTED!"
        color = (0, 0, 255)
    elif is_accident:
        label = "ACCIDENT DETECTED!"
        color = (0, 165, 255)
    else:
        label = "Normal"
        color = (0, 255, 0)

    cv2.putText(annotated, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Accident Detection System", annotated)

    # ESC press chesthe stop avutundi
    if cv2.waitKey(1) & 0xFF == 27:
        print("[STOPPED] User stopped detection.")
        break

cap.release()
cv2.destroyAllWindows()
