import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR = r"aug"

TEST_SIZE = 0.10
VAL_SIZE = 0.10

RANDOM_STATE = 42


def extract_label_from_path(image_path, dir_position=-2, separator="/"):
    return image_path.split(separator)[dir_position]


def save_as_csv(images, labels, file_name):
    df = pd.DataFrame({"image_path": images, "labels": labels})
    csv_file_path = os.path.join(BASE_DIR, file_name)
    df.to_csv(csv_file_path, index=False)


images = glob.glob(f"{BASE_DIR}/**/*.jpg")
# aug/Mild/aug-_0_3489125.jpg
labels = [extract_label_from_path(image_path) for image_path in images]

# test split
X_, X_test, y_, y_test = train_test_split(
    images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)

# valid split
X_train, X_val, y_train, y_val = train_test_split(
    X_, y_, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_
)

if __name__=="__main__":
    #save to csvs

    save_as_csv(X_train, y_train, "train_aug.csv")
    save_as_csv(X_val, y_val, "val_aug.csv")
    save_as_csv(X_test, y_test, "test_aug.csv")

    print("CSVs files  generated successfully.")

    plt.hist(y_test)
    plt.title("Test data distribution1")
    plt.savefig("data/Test_data_distribution1.png")
    plt.show()
