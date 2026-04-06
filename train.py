import torch
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    device=0 if torch.cuda.is_available() else 'cpu',
    batch=-1,  # auto batch
    patience=50,
    save_period=10,
    project="runs/detect",
    name="accident_cnn",
    pretrained=True,
    verbose=True
)
