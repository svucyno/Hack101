# extract_frames.py
# Video nundi frames extract chesi dataset/images/train lo save chestham

import cv2
import os

video_path = "input.mp4"
output_dir = "data/train/extracted"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Video open cheyaledu!")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Total frames: {total_frames} | FPS: {fps:.1f}")

# Every 5th frame save chestham (too many frames avoid cheyyadaniki)
save_every = 5
saved = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % save_every == 0:
        img_name = f"frame_{frame_count:05d}.jpg"
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, frame)
        saved += 1

cap.release()
print(f"[DONE] {saved} frames saved to '{output_dir}'")
print(f"[NEXT] Ippudu Roboflow lo upload cheyandi!")
