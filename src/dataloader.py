from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, TEST_CSV_PATH, TRAIN_CSV_PATH, VAL_CSV_PATH
from src.dataset import DRDataset
from src.transforms import transform


train_dataset = DRDataset(csv_path=TRAIN_CSV_PATH, transforms=transform)
val_dataset = DRDataset(csv_path=VAL_CSV_PATH, transforms=transform)
test_dataset = DRDataset(csv_path=TEST_CSV_PATH, transforms=transform)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    images, labels = next(iter(train_data_loader))

    print(images, labels)
