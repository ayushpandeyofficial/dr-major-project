import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR = r"data"

TEST_SIZE = 0.15
VAL_SIZE = 0.1

RANDOM_STATE = 42


def extract_label_from_path(image_path, dir_position=-2, separator="/"):
    return image_path.split(separator)[dir_position]


def save_as_csv(images, labels, file_name):
    df = pd.DataFrame({"image_path": images, "labels": labels})
    csv_file_path = os.path.join(BASE_DIR, file_name)
    df.to_csv(csv_file_path, index=False)


images = glob.glob(f"{BASE_DIR}/**/*.png")
labels = [extract_label_from_path(image_path) for image_path in images]

# test split
X_, X_test, y_, y_test = train_test_split(
    images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)

# valid split
X_train, X_val, y_train, y_val = train_test_split(
    X_, y_, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_
)


# save to csvs

save_as_csv(X_train, y_train, "train.csv")
save_as_csv(X_val, y_val, "val.csv")
save_as_csv(X_test, y_test, "test.csv")

print("CSVs files  generated successfully.")
