import numpy as np

class SegmentationModel:
    def __init__(self):
        self.classes = ['Urban', 'Vegetation', 'Water']
        
    def predict(self, image_array):
        """
        Simulates segmentation by thresholding color channels.
        Input: image_array (H, W, 3)
        Output: mask (H, W) with class indices
        """
        # Heuristic: 
        # Green dominant -> Vegetation (1)
        # Blue dominant -> Water (2)
        # Else -> Urban (0)
        
        red_channel = image_array[:, :, 0]
        green_channel = image_array[:, :, 1]
        blue_channel = image_array[:, :, 2]
        
        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        
        # Vegetation: Green > Red and Main Component
        vegetation_mask = (green_channel > red_channel) & (green_channel > blue_channel)
        mask[vegetation_mask] = 1
        
        # Water: Blue > Red and Blue > Green
        water_mask = (blue_channel > red_channel) & (blue_channel > green_channel)
        mask[water_mask] = 2
        
        return mask
