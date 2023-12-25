import os
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.config import EPOCHS, LR, SEED
from src.dataloader import train_data_loader, val_data_loader
from src.io import save_model_checkpoint
from src.models.resnet50 import ResNet50Model

from visualize_graph import visualize_graph

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device:{device}")

torch.manual_seed(SEED)

best_val_acc = 0

dt = datetime.now()
f_dt = dt.strftime("%Y-%m-%d-%H-%M-%S")
folder_name = f"resnet50_run-{f_dt}"
os.mkdir(f"artifacts/{folder_name}")

writer = SummaryWriter(log_dir=f"artifacts/{folder_name}/tensorboard_logs")

# Using  ResNet50 model
model = ResNet50Model(num_labels=5)

model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

epochwise_train_loss = []
epochwise_val_loss = []
epochwise_train_acc = []
epochwise_val_acc = []

for epoch in range(EPOCHS):
    train_running_loss = 0
    val_running_loss = 0
    train_running_acc = 0
    val_running_acc = 0

    # Training
    model.train()
    for i, (images, labels) in enumerate(train_data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        model_out = model(images)

        labels = labels.to(model_out.device)
        loss = criterion(model_out, labels)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = torch.argmax(model_out, dim=1)
        acc = (pred == labels).float().mean()
        train_running_acc += acc.item()

    # Validation during training (nested loop)
    model.eval()
    val_running_loss = 0
    val_running_acc = 0

    for val_images, val_labels in val_data_loader:
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        val_model_out = model(val_images)
        val_labels = val_labels.to(val_model_out.device)
        val_loss = criterion(val_model_out, val_labels)
        val_running_loss += val_loss.item()
        val_pred = torch.argmax(val_model_out, dim=1)
        val_acc = (val_pred == val_labels).float().mean()
        val_running_acc += val_acc.item()

    avg_train_loss = train_running_loss / len(train_data_loader)
    avg_val_loss = val_running_loss / len(val_data_loader)

    avg_val_acc = val_running_acc / len(val_data_loader)
    avg_train_acc = train_running_acc / len(train_data_loader)

    epochwise_train_loss.append(avg_train_loss)
    epochwise_val_loss.append(avg_val_loss)

    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Loss/test", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_train_acc, epoch)
    writer.add_scalar("Accuracy/test", avg_val_acc, epoch)

    epochwise_train_acc.append(avg_train_acc)
    epochwise_val_acc.append(avg_val_acc)

    print(
        f"Epoch {epoch} Train Loss: {avg_train_loss:.3f} Val Loss : {avg_val_loss:.3f} \t "
        f"Train Accuracy : {avg_train_acc:.3f} \t  Validation Accuracy : {avg_val_acc:.3f}"
    )

    best_val_acc, checkpoint_path, best_model_path = save_model_checkpoint(
        model,
        optimizer,
        folder_name,
        EPOCHS,
        avg_train_loss,
        avg_val_loss,
        avg_train_acc,
        avg_val_acc,
        best_val_acc,
    )
visualize_graph(epochwise_train_acc,epochwise_val_acc,epochwise_train_loss,epochwise_val_loss,folder_name)