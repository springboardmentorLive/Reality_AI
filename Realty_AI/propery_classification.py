import cv2
import numpy as np
from sklearn.svm import SVC

# Training data (features must be in a 2D array)
X_train = [[100], [150], [200]] 
y_train = ["Old", "Average", "New"]  # Target labels

# Initialize and train the model
model = SVC()
model.fit(X_train, y_train)

# Test image: '8' loads as reduced grayscale
img = cv2.imread("data/images/house.jpg", 8)

# Check if image was loaded correctly
if img is not None:
    # Calculate average brightness
    brightness = int(np.mean(img))
    
    # Predict: Input must be a 2D array [[value]]
    prediction = model.predict([[brightness]])
    
    print("Property Condition:", prediction[0])