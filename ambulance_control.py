import time
from pathlib import Path

import cv2
import serial
from serial import SerialException
from ultralytics import YOLO

COM_PORT = "COM3"
BAUD_RATE = 9600
VIDEO_SOURCE = "ambulance.mp4"  # Change to 0 for webcam
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

# The default YOLOv8 COCO model does not contain a real "ambulance" class.
# Until you train a custom model, bus/truck are used as the emergency fallback.
PRIORITY_CLASSES = {"ambulance", "bus", "truck"}


def connect_arduino():
    try:
        board = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Arduino connected on {COM_PORT}")
        return board
    except SerialException as exc:
        print(f"⚠️ Arduino not available on {COM_PORT}: {exc}")
        print("Running in preview mode without sending serial data.")
        return None


def send_signal(board, value: str):
    if board is None:
        return

    try:
        board.write(value.encode("utf-8"))
    except SerialException as exc:
        print(f"⚠️ Failed to send signal to Arduino: {exc}")


def is_priority_vehicle(class_name: str, confidence: float) -> bool:
    return class_name.lower() in PRIORITY_CLASSES and confidence >= CONFIDENCE_THRESHOLD


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    board = connect_arduino()
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    green_signal_on = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ambulance_detected = False
            results = model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = str(model.names[class_id])

                    if is_priority_vehicle(class_name, confidence):
                        ambulance_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(
                            frame,
                            f"AMBULANCE PRIORITY {confidence:.2f}",
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

            if ambulance_detected and not green_signal_on:
                send_signal(board, "1")
                green_signal_on = True
                print("🟢 Ambulance detected -> GREEN signal sent")
            elif not ambulance_detected and green_signal_on:
                send_signal(board, "0")
                green_signal_on = False
                print("🔴 Ambulance cleared -> NORMAL signal sent")

            status_text = "GREEN SIGNAL" if ambulance_detected else "NORMAL MODE"
            status_color = (0, 255, 0) if ambulance_detected else (0, 0, 255)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            cv2.circle(frame, (500, 35), 12, status_color, -1)

            cv2.imshow("Ambulance Traffic Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        if green_signal_on:
            send_signal(board, "0")
        if board is not None and board.is_open:
            board.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()