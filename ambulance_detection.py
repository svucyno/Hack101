import cv2
import serial
import time
import numpy as np
from ultralytics import YOLO

# ---------------- Arduino ----------------
arduino = serial.Serial('COM3', 9600)
time.sleep(2)

# ---------------- Model ----------------
model = YOLO("yolov8n.pt")

# ---------------- Video ----------------
cap = cv2.VideoCapture("ambulance.mp4")

last_signal = None  # prevent spamming Arduino

print("🚦 Ambulance Detection Started")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    ambulance_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # YOLO doesn't have ambulance → simulate using bus/truck
            if label.lower() in ["truck", "bus"]:
                ambulance_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(frame, "AMBULANCE", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ---------------- SIGNAL LOGIC ----------------
    if ambulance_detected:
        signal = "GREEN"
        color = (0,255,0)

        if last_signal != "GREEN":
            arduino.write(b'A')
            print("🟢 GREEN SENT")
            last_signal = "GREEN"

    else:
        signal = "RED"
        color = (0,0,255)

        if last_signal != "RED":
            arduino.write(b'N')
            print("🔴 RED SENT")
            last_signal = "RED"

    # ---------------- DISPLAY ----------------
    display = frame.copy()

    cv2.rectangle(display, (20,20), (300,80), color, -1)
    cv2.putText(display, signal, (50,65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

    cv2.imshow("Ambulance Detection", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()