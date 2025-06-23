# Lettuce Growth AI for Vertical Farming

This project aims to develop an Artificial Intelligence system to support lettuce growth optimization in vertical farming environments. The AI components focus on detecting diseases, classifying growth stages, and predicting yield using image data and features.

---

## Project Overview

Vertical farming offers efficient, sustainable food production. This project applies computer vision and machine learning techniques to analyze lettuce images and relevant features, enabling:

- **Disease Detection:** Identify common diseases on lettuce leaves using object detection models.
- **Growth Stage Classification:** Classify lettuce plants into different growth stages using deep learning image classification.
- **Yield Prediction:** Estimate the expected lettuce yield based on environmental and growth features using machine learning regression.

---

## Current Components

### 1. Dataset Setup (`dataset_setup.py`)

- Downloads the latest lettuce image dataset from Roboflow.
- Prepares the dataset in YOLOv8 format for training.
- Displays a sample image for quick verification.

### 2. Disease Detection (`detect_disease.py`)

- Trains a YOLOv8 model to detect diseases on lettuce leaves.
- Runs inference on validation images and saves output visualizations.

### 3. Growth Stage Classification (`stage_classifier.py`)

- Uses a pretrained ResNet18 model fine-tuned on labeled growth stage images.
- Classifies images into growth stages with a training loop and loss reporting.

### 4. Yield Prediction (`yield_prediction.py`)

- Trains a Random Forest regression model on tabular feature data to predict lettuce yield.
- Reports Mean Absolute Error (MAE) to measure prediction accuracy.

### 5. Model Training (`train_yolo.py`)

- Provides a simple script to train the YOLOv8 model with configurable parameters such as epochs, batch size, and image size.

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_folder>

   ```

2. **Setup Python environment**

   Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run dataset setup**
   Execute the dataset setup script to download and prepare the dataset:

   ```bash
   python dataset_setup.py
   ```

4. **Train models and run inference**
   - For disease detection:
     ```bash
     python detect_disease.py
     ```
   - For growth stage classification:
     ```bash
     python stage_classifier.py
     ```
   - For yield prediction:
     ```bash
     python yield_prediction.py
     ```

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- ultralytics (for YOLOv8)
- scikit-learn
- OpenCV
- matplotlib
- Roboflow Python package

## Future Work

- Improve dataset quality and size.
- Integrate environmental sensor data for more accurate yield prediction.
- Develop real-time monitoring with camera feeds.
- Explore additional ML models and hyperparameter tuning.
- Deploy models to edge devices for on-site vertical farm monitoring.

## Contact

For questions or contributions, please contact me at [joshuajones272000@gmail.com]
