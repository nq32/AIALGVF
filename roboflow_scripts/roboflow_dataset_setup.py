# Import necessary libraries
from roboflow import Roboflow  # For accessing Roboflow datasets
import cv2                     # For image processing
import os                      # For file and directory operations
import matplotlib.pyplot as plt # For displaying images

# --- Roboflow setup ---
print("loading Roboflow workspace...")  # Notify user of workspace loading
rf = Roboflow(api_key="9G3qcy2ayjhW8p7r49br")  # Initialize Roboflow with API key
workspace = rf.workspace("aialgvf")            # Access specific workspace
print("loading Roboflow project...")           # Notify user of project loading
project = workspace.project("lettuce-zj97g-bclgm")  # Access specific project

# --- Check versions and pick the latest one ---
versions = list(project.versions())  # Get all available versions
if not versions:
    raise ValueError("No versions found for this project.")

latest_version = max(versions, key=lambda v: v.version)  # Find the version with the highest number
print(f"Latest version: {latest_version.version} - {latest_version.name}")

print(f"Downloading dataset version {latest_version.version} ...")
dataset = project.version(latest_version.version).download("yolov8", location="dataset")

# --- Load and show a sample image ---
# Set path to image directory
image_dir = dataset.location + "/train/images"  # Path to training images

# List all .jpg files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]  # Get list of image files

if not image_files:
    raise ValueError(f"No .jpg images found in directory: {image_dir}")  # Check if image list is empty

# Get the first image file
first_image_path = os.path.join(image_dir, image_files[0])  # Path to first image
img = cv2.imread(first_image_path)                          # Read image

# Validate the image
if img is None:
    raise ValueError(f"Could not load image: {first_image_path}")  # Raise error if image not loaded

# Convert to RGB and display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
plt.imshow(img_rgb)                             # Display image
plt.title("Sample Lettuce Image")               # Set plot title
plt.axis("off")                                # Hide axes
plt.show()                                     # Show plot
