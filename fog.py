import os
import cv2
import numpy as np

def add_fog_effect(image, intensity=1.0, fog_color=(240, 240, 245)):
    """
    Adds a fog effect to an image.
    :param image: Input image
    :param intensity: Intensity of the fog, range [0, 1]
    :param fog_color: Color of the fog, default is light grey, adjusted for visibility in daylight
    :return: Image with fog effect
    """
    height, width = image.shape[:2]
    fog = np.zeros((height, width, 3), dtype=np.uint8)
    fog[:] = fog_color
    noise_scale = 10  # Larger scale for more impactful noise
    x = np.linspace(0, noise_scale, width, endpoint=False)
    y = np.linspace(0, noise_scale, height, endpoint=False)
    x, y = np.meshgrid(x, y)
    z = (np.sin(x) + np.cos(y)) * 10  # Modified to increase pattern variance
    noise = cv2.normalize(z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fog[:, :, :] = noise[:, :, None]
    fog = cv2.GaussianBlur(fog, (51, 51), 0)  # Larger blur for smoother transition
    fogged_image = cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)
    return fogged_image

def process_images(source_dir, target_dir, intensity=0.6):
    """
    Processes all images in the source directory, applies fog effect,
    and saves them to the target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                foggy_image = add_fog_effect(image, intensity=intensity)
                foggy_img_path = os.path.join(target_dir, filename)
                cv2.imwrite(foggy_img_path, foggy_image)
                print(f"Saved foggy image to {foggy_img_path}")
            else:
                print(f"Failed to load image at {img_path}")

# Define source and target directories
source_directory = r'C:\Users\nikhil\Desktop\CS412\Fiji Data\Augmented\Day\5'
target_directory = r'C:\Users\nikhil\Desktop\CS412\Fiji Data\Night Augmented - Fog\Day\5'

# Process all images with a specified intensity
process_images(source_directory, target_directory, intensity=0.6)
