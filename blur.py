#https://www.kaggle.com/code/pritishmishra/augmentation-with-albumentations
#The above link is used for the following blur effect code


import os
import albumentations as A
import cv2
import numpy as np

# Directories
BASE_DIR = "C:\\Users\\nikhil\\Desktop\\CS412\\Fiji Data\\Augmented\\Night\\1"
OUTPUT_DIR = "C:\\Users\\nikhil\\Desktop\\CS412\\Fiji Data\\Augmented - Blurred\\Night\\1"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Augmentation Pipeline using Gaussian Blur
transform = A.Compose([
    A.GaussianBlur(blur_limit=(17, 19), sigma_limit=0, p=1)  # Adjust `blur_limit` and `sigma_limit` as needed
])


# Function to load and convert image
def load_img(path):
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to save image
def save_img(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

# Process all images in the directory
for filename in os.listdir(BASE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(BASE_DIR, filename)
        image = load_img(image_path)
        if image is not None:
            # Apply transformation
            transformed_img = transform(image=image)['image']
            # Save transformed image
            output_path = os.path.join(OUTPUT_DIR, filename)
            save_img(transformed_img, output_path)
            print(f'Processed and saved: {filename}')
        else:
            print(f'Failed to load: {filename}')
    else:
        print(f'Skipped non-image file: {filename}')
