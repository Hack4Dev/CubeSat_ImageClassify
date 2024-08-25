import os
import numpy as np
from PIL import Image

def read_images_from_folder(folder_path):
    images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # Check for common image file extensions
            img_path = os.path.join(folder_path, filename)

            with Image.open(img_path) as img:
                img_array = np.array(img, dtype=np.uint8)  # Convert image to NumPy array with float32here, and the code be
                images.append(img_array)  # Add the image array to the list

    images_np = np.stack(images)  # Convert the list of image arrays to a single NumPy array
    return images_np