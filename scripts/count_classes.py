# import os

# data_dir = "data" 
# folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# for folder in folders:
#     folder_path = os.path.join(data_dir, folder)
#     images = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
#     num_images = len(images)
#     print(f"Folder '{folder}' contains {num_images} image(s).")
    

import pandas as pd

# Read the train1.csv file
train_df = pd.read_csv("data/train1.csv")

# Count the number of images for each label
label_counts = train_df['labels'].value_counts()

# Print the counts
print("Number of images for each label in train1.csv:")
for label, count in label_counts.items():
    print(f"{label}: {count} images")
