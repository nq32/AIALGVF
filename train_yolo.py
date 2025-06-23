from ultralytics import YOLO  # Import the YOLO class from the ultralytics library

# Load a YOLOv8 model – you can change to 'yolov8s.pt', 'yolov8m.pt', etc.
model = YOLO("yolov8n.pt")  # Load the YOLOv8 nano model for faster testing

# Train the model
model.train(
    data="lettuce-1/data.yaml",  # Path to the dataset configuration file
    epochs=50,                   # Number of training epochs
    imgsz=640,                   # Image size for training
    batch=8,                     # Batch size (adjust based on GPU memory)
    project="runs",              # Directory to save training results
    name="lettuce_train",        # Name of the training run
    exist_ok=True,               # Overwrite existing run if it exists
    device="0",                  # Specify the GPU device (e.g., "0" for the first GPU, "cpu" for CPU training)
)
