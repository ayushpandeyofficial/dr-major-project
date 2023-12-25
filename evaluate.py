import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from src.config import BEST_RESNET50_MODEL
from src.dataloader import test_data_loader
from src.models.resnet50 import ResNet50Model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNet50Model(num_labels=5)
model.load_state_dict(torch.load(BEST_RESNET50_MODEL))
model = model.to(device)
model.eval()

# evaluate the model
all_preds = []
all_labels = []

for test_images, test_labels in test_data_loader:
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    test_model_out = model(test_images)
    test_labels = test_labels.to(test_model_out.device)
    test_pred = torch.argmax(test_model_out, dim=1)
    
    all_preds.extend(test_pred.cpu().numpy())
    all_labels.extend(test_labels.cpu().numpy())



# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds)

# Plot the confusion matrix with annotations
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"],
    yticklabels=["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(class_report)
