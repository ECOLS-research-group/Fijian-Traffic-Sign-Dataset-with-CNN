import cv2
import os
import numpy as np

def apply_motion_blur(image, size, angle):
    """
    Applies motion blur to an image.
    :param image: Input image
    :param size: Size of the motion blur (intensity)
    :param angle: Angle of the motion blur in degrees
    :return: Image with motion blur applied
    """
    # Create the motion blur kernel
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size

    # Rotate the kernel by the given angle
    rotation_matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))

    # Apply the kernel to the image
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image

def process_images(source_dir, target_dir, blur_size, blur_angle):
    """
    Processes all images in the source directory, applies motion blur,
    and saves them to the target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                blurred_image = apply_motion_blur(image, blur_size, blur_angle)
                blurred_img_path = os.path.join(target_dir, filename)
                cv2.imwrite(blurred_img_path, blurred_image)
                print(f"Saved motion-blurred image to {blurred_img_path}")
            else:
                print(f"Failed to load image at {img_path}")

# Define source and target directories
source_directory = r'C:\Users\nikhil\Desktop\CS412\Fiji Data\Normal\Day\StopSign'
target_directory = r'C:\Users\nikhil\Downloads\New Folder'

# Apply motion blur with a size of 15 and an angle of 30 degrees
process_images(source_directory, target_directory, blur_size=5, blur_angle=180)
