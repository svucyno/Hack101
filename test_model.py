from ultralytics import YOLO

model = YOLO("tvd_model_best.pt")

results = model.predict(source="test.jpg", show=True)

print("Done")