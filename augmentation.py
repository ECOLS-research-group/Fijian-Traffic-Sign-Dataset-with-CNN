import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img

# Directory paths
input_dir = "C:\\Users\\nikhil\\Desktop\\CS412\\Fiji Data\\Normal\\Day\\StopSign"  # Path to original images
output_dir = "C:\\Users\\nikhil\\Desktop\\CS412\\Fiji Data\\Augmented Data\\Day\\6"  # Path where augmented images will be saved

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Process each image
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):  # Check for image format
        image_path = os.path.join(input_dir, filename)
        img = load_img(image_path)  # Load image
        x = img_to_array(img)  # Convert image to numpy array
        x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)
        
        # Generate and save 10 augmented images for each input image
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 18:
                break  # Stop after generating 10 images

print("Augmentation complete.")
