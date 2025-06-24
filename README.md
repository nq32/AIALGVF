# Lettuce Growth AI for Vertical Farming

This project develops AI tools to optimize lettuce growth in vertical farming environments. It leverages computer vision and machine learning for disease detection, growth stage classification, yield prediction, and health status classification using image data.

---

## Project Overview

- **Disease Detection:** Identify common diseases on lettuce leaves using object detection (YOLOv8).
- **Growth Stage Classification:** Classify lettuce plants into growth stages with deep learning.
- **Yield Prediction:** Estimate expected lettuce yield from environmental and growth features.
- **Health Status Classification:** Classify lettuce as healthy or unhealthy using deep learning models trained on Kaggle and Roboflow datasets.

---

## Current Components

### 1. Dataset Setup (`kaggle_dataset_setup.py`, `roboflow_dataset_setup.py`)

- Downloads and prepares lettuce image datasets from Kaggle and Roboflow.
- Preserves original folder structures and ignores unsupported file types.

### 2. Disease Detection (`kaggle_detect_disease.py`, `roboflow_detect_disease.py`)

- Trains a YOLOv8 model to detect diseases on lettuce leaves.
- Runs inference and saves output visualizations.

### 3. Growth Stage Classification (`stage_classifier.py`)

- Fine-tunes a ResNet18 model on labeled growth stage images.
- Classifies images into growth stages.

### 4. Yield Prediction (`yield_prediction.py`)

- Trains a Random Forest regression model on tabular data to predict lettuce yield.
- Reports Mean Absolute Error (MAE).

### 5. Health Status Classification (`kaggle_detect_status.py`, `test.py`)

- Trains a MobileNetV2-based classifier to distinguish healthy vs. unhealthy lettuce using images from nested folders.
- Supports custom folder structures.
- Evaluates the trained model on new datasets (e.g., Roboflow) and outputs predictions and scores to a CSV file.

### 6. Model Training (`train_yolo.py`)

- Script to train the YOLOv8 model with configurable parameters.

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Setup Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run dataset setup**

   ```bash
   python kaggle_scripts/kaggle_dataset_setup.py
   python roboflow_scripts/roboflow_dataset_setup.py
   ```

4. **Train models and run inference**

   - Disease detection:
     ```bash
     python detect_disease.py
     ```
   - Growth stage classification:
     ```bash
     python stage_classifier.py
     ```
   - Yield prediction:
     ```bash
     python yield_prediction.py
     ```
   - Health status classification:
     ```bash
     python kaggle_scripts/kaggle_detect_status.py
     ```
   - Test health classifier on new images:
     ```bash
     python test.py
     ```

---

## Dependencies

- Python 3.8+
- TensorFlow
- PyTorch
- torchvision
- ultralytics (for YOLOv8)
- scikit-learn
- OpenCV
- matplotlib
- Roboflow Python package
- kagglehub

---

## Future Work

- Improve dataset quality and size.
- Integrate environmental sensor data for more accurate yield prediction.
- Real-time monitoring with camera feeds.
- Explore additional ML models and hyperparameter tuning.
- Deploy models to edge devices for on-site vertical farm monitoring.

---

## Contact

For questions or contributions, please contact [joshuajones272000@gmail.com]
