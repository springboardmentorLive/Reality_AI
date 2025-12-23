import cv2
import matplotlib.pyplot as plt

# 1. Read the image
img = cv2.imread("data/images/house.jpg")

# 2. Convert to grayscale (Fix: cvtColor instead of evtColor)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Simple segmentation (Fix: Unpack tuple, fix constant underscore)
ret, segmented = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# 4. Create display figure
plt.figure(figsize=(8,4))

# Subplot 1: Original Image (Fix: Convert BGR to RGB for matplotlib)
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

# Subplot 2: Segmented Image (Fix: Display unpacked 'segmented' variable)
plt.subplot(1,2,2)
plt.title("Segmented Image")
plt.imshow(segmented, cmap="gray")

# 5. Display the result (Fix: plt instead of pls)
plt.show()