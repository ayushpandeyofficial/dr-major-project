import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, TRAIN_CSV_PATH, VAL_CSV_PATH
from src.dataset import DRDataset
from src.transforms import train_transform, val_transform
from src.utilis import label_to_idx

train_dataset = DRDataset(csv_path=TRAIN_CSV_PATH, transforms=train_transform)
train_labels = torch.tensor(
    [label_to_idx(label) for label in train_dataset.labels], dtype=torch.long
)
val_dataset = DRDataset(csv_path=VAL_CSV_PATH, transforms=val_transform)
val_label= torch.tensor(
    [label_to_idx(label) for label in val_dataset.labels], dtype=torch.long
)
# test_dataset = DRDataset(csv_path=TEST_CSV_PATH, transforms=val_transform)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
if __name__ == "__main__":
    print(train_labels)

    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    y = train_labels.numpy()
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    print("Weights", weights)


# images, labels = next(iter(train_data_loader))

# print(images, labels)
# from torch.utils.data import DataLoader, WeightedRandomSampler

# from src.config import BATCH_SIZE, TEST_CSV_PATH, TRAIN_CSV_PATH, VAL_CSV_PATH
# from src.dataset import DRDataset
# from src.transforms import train_transform,val_transform
# from src.utilis import label_to_idx


# # Assuming your class distribution dictionary is defined as follows
# class_distribution = {
#     'No_DR':1461,
#     'Mild': 300,
#     'Moderate1':381,
#     'Proliferate_DR':238,
#     'Severe': 157
# }
# train_dataset = DRDataset(csv_path=TRAIN_CSV_PATH, transforms=train_transform)
# val_dataset = DRDataset(csv_path=VAL_CSV_PATH, transforms=val_transform)
# test_dataset = DRDataset(csv_path=TEST_CSV_PATH, transforms=val_transform)

# #train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# labels = train_dataset.labels

# # Calculate weights for each class to balance the dataset
# class_weights = [1.0 / class_distribution[key] for key in class_distribution.keys()]
# sample_weights = [class_weights[label_to_idx(label)] for label in labels]

# # Create a WeightedRandomSampler
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# # Use the sampler in the DataLoader for training
# train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

# if __name__ == "__main__":
#     images, labels = next(iter(train_data_loader))

#     print(images, labels)
