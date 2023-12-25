import torch
from src.config import BEST_RESNET50_MODEL,TRAIN_CSV_PATH,VAL_CSV_PATH,TEST_CSV_PATH
import matplotlib.pyplot as plt 
from src.models.resnet50 import ResNet50Model
import pandas as pd


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ResNet50Model(num_labels=5)
model.load_state_dict(torch.load(BEST_RESNET50_MODEL))
model = model.to(device)
model.eval()

# Load training and validation metrics from CSV files
train_metrics = pd.read_csv(TRAIN_CSV_PATH)
test_metrics = pd.read_csv(TEST_CSV_PATH)












# # Plotting the loss over epochs
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_metrics['epoch'], train_metrics['train_loss'], label="Train Loss", marker='o')
# plt.plot(test_metrics['epoch'], test_metrics['test_loss'], label="Test Loss", marker='o')
# plt.title("Loss Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# # Plotting the accuracy over epochs
# plt.subplot(1, 2, 2)
# plt.plot(train_metrics['epoch'], train_metrics['train_acc'], label="Train Accuracy", marker='o')
# plt.plot(test_metrics['epoch'], test_metrics['test_acc'], label="Test Accuracy", marker='o')
# plt.title("Accuracy Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.tight_layout()
# plt.show()