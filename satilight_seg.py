import cv2
import matplotlib.pyplot as plt

# 1. Read satellite image
img = cv2.imread("images\OIP (6).jpg")   # change to your satellite image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Apply Gaussian Blur (important for satellite images)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Apply Otsu Thresholding
_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 5. Display results
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title("Original Satellite Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grayscale Image")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmented Image")
plt.imshow(binary, cmap="gray")
plt.axis("off")

plt.show()