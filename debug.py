# debug.py
# Idi run cheyandi - video lo em detect avutundо chustam

from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video open cheyaledu!")
    exit()

frame_count = 0
print("=== DETECTION DEBUG ===\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Every 10th frame matrame check chestham (speed up)
    if frame_count % 10 != 0:
        continue

    results = model(frame, verbose=False)

    print(f"--- Frame {frame_count} ---")
    found_anything = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = model.names[cls_id]
            print(f"  Detected: {name} (cls={cls_id}) | Confidence: {conf:.2f}")
            found_anything = True

    if not found_anything:
        print("  Nothing detected.")

    if frame_count >= 100:  # First 100 frames chalu
        break

cap.release()
print("\n=== DEBUG COMPLETE ===")
