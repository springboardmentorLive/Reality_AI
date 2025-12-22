import random

class ConditionModel:
    def __init__(self):
        self.conditions = ['New', 'Moderate', 'Old']
        
    def predict(self, image_array):
        """
        Simulates condition classification.
        Returns a random condition and a confidence score.
        """
        # In a real model, this would forward pass through a generic CNN
        prediction = random.choice(self.conditions)
        confidence = random.uniform(0.7, 0.99)
        return prediction, confidence
