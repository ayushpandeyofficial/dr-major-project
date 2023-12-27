from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


image_path = r'data/Proliferate_DR/0ada12c0e78f.png'
original_image = Image.open(image_path)

# Define the transformations
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783,0.0783,0.0783]),
])

# Apply the transformations
transformed_image = transforms_train(original_image)

# Convert the transformed image tensor to a NumPy array
transformed_image_np = transformed_image.numpy()

# Rescale values to [0, 1]
transformed_image_np = (transformed_image_np + 1) / 2.0

# Display the original and transformed images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(np.asarray(original_image))
axes[0].set_title('Original Image')

# Transformed image
axes[1].imshow(np.transpose(transformed_image_np, (1, 2, 0)))
axes[1].set_title('Transformed Image')

plt.show()

# transforms_val = transforms.Compose([
#         transforms.Resize((256, 256)),

#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
# # Apply the transformations
# transformed_image1 = transforms_val(original_image)

# # Convert the transformed image tensor to a NumPy array
# transformed_image_np1 = transformed_image1.numpy()

# # Rescale values to [0, 1]
# transformed_image_np1 = (transformed_image_np1 + 1) / 2.0

# # Display the original and transformed images
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # Original image
# axes[0].imshow(np.asarray(original_image))
# axes[0].set_title('Original Image')

# # Transformed image
# axes[1].imshow(np.transpose(transformed_image_np1, (1, 2, 0)))
# axes[1].set_title('Transformed Image')

# plt.show()