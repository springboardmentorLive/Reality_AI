import cv2
import numpy as np 
from sklearn.svm import svc

x_train = [[100],[150],[200]]
y_train = ["new","Average","old"]

model = svs()
model.fit(x_train,y_train)

img = cv2.imread("images\OIP (6).jpg" , 0)
brightness  = int(np.mean(img))

prediction = model.predict ([[brightness]])
print("Property condition:" , prediction[0])