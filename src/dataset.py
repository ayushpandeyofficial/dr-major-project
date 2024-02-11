# 1 Prepare dataset
from torch.utils.data import Dataset
from src.utilis import read_as_csv, label_to_idx
from PIL import Image
import os

class DRDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        images, labels = read_as_csv(csv_path)
        self.images = images
        self.labels = labels
        self.transforms = transforms
    

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return f"ImageDataset with {self.__len__()} samples"

# if there is no full path in train.csv 
    # def __getitem__(self, index):
    #     image_dir = os.path.join("challenge_data", "train_resized")
    #     image_name = self.images[index]  # Get the image name without extension
    #     label_name = self.labels[index]
    #     label = label_to_idx(label_name)

    #         # Append a default extension based on known formats
    #     valid_extensions = {'.jpg', '.jpeg', '.png'}
    #     image_extension = next(ext for ext in valid_extensions if os.path.exists(os.path.join(image_dir, f"{image_name}{ext}")))
    #     image_path = os.path.join(image_dir, f"{image_name}{image_extension}")

    #     # Check if the file exists before trying to open it
    #     if not os.path.exists(image_path):
    #         print(f"Error: File not found - {image_path}")
    #         return None, None

    #     image = Image.open(image_path).convert("RGB")

    #     if self.transforms:
    #         image = self.transforms(image)

    #     return image, label

# if there is full path in train.csv 
    def __getitem__(self, index):
        image_path = self.images[index]  # Get the image name without extension
        label_name = self.labels[index]
        label = label_to_idx(label_name)

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, label

