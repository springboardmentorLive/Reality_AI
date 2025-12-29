import cv2
import matplotlib.pyplot as plt

# Image ko load karein
img = cv2.imread("images/house.png")
# Image ko gray color mein convert karein
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simple segmentation using thresholding
# Ye 120 se zyada brightness wale pixels ko white (255) kar dega
_, segmented = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Results ko display karein
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented, cmap="gray")

plt.show()