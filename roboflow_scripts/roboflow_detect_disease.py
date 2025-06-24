from ultralytics import YOLO
from roboflow_scripts.roboflow_dataset_setup import dataset

# Load YOLOv8 model (train from scratch or fine-tune)
model = YOLO("yolov8n.pt")  # or yolov8s/m for faster training

# Train
model.train(data=dataset.location + "/data.yaml",
            epochs=20,
            batch=16,
            imgsz=640,
            device=0)

# Inference example
results = model.predict(source=dataset.location + "/valid/images", imgsz=640, save=True)
results[0].show()
