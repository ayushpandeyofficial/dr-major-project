# 1 Prepare dataset
from PIL import Image
from torch.utils.data import Dataset

from src.utilis import label_to_idx, read_as_csv


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

    def __getitem__(self, index):
        image_path = self.images[index]
        label_name = self.labels[index]
        label = label_to_idx(label_name)

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, label

