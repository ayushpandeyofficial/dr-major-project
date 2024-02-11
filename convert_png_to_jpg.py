from PIL import Image
import os

# Set the directories
input_dir = 'data/No_DR'
output_dir = 'aug/No_DR'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all files in the input directory
files = os.listdir(input_dir)

# Iterate through each file
for file in files:
    if file.endswith('.png'):
        # Open the image
        image_path = os.path.join(input_dir, file)
        img = Image.open(image_path)
        
        # Remove the file extension and replace it with .jpg
        new_filename = os.path.splitext(file)[0] + '.jpg'
        
        # Save the image in the output directory
        output_path = os.path.join(output_dir, new_filename)
        img.save(output_path, 'JPEG')
