# stage_classifier.py
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Device configuration: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transformations: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load training and validation datasets
train_data = datasets.ImageFolder("data/stage/train", transform=transform)
val_data = datasets.ImageFolder("data/stage/val", transform=transform)  # Optional validation folder

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Define number of classes based on folder names
num_classes = len(train_data.classes)
print(f"Number of classes: {num_classes}")

# Load pretrained ResNet18 and modify final layer for our classes
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f}")

    # Validation (optional)
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"           — Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

print("Training complete.")
