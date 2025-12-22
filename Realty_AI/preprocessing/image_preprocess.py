import numpy as np
from PIL import Image

def load_and_preprocess_image(image_file, target_size=(256, 256)):
    """
    Loads an image file, resizes it, and returns a numpy array.
    """
    try:
        image = Image.open(image_file)
        image = image.resize(target_size)
        image_array = np.array(image)
        # Normalize to 0-1 range if needed for deep learning (omitted for simple visualization)
        return image, image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None
