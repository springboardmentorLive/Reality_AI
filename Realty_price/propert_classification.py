import cv2
import numpy as np
from sklearn.svm import SVC

# Dummy feature-based training
X_train = [[100], [150], [200]] # image brightness
y_train = ["New", "Average", "Old"]

model = SVC()
model.fit(X_train, y_train)

# Test image logic (Ensure images/house.png exists)
try:
    img = cv2.imread("images/house.png", 0)
    brightness = int(np.mean(img))
    prediction = model.predict([[brightness]])
    print("Property Condition:", prediction[0])
except:
    print("Image not found at images/house.png")