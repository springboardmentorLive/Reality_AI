import cv2
import matplotlib.pyplot as plt  # Fixed alias from 'pid' to 'plt'

# 1. Load the image
img = cv2.imread("images/house.jpg")

# 2. Convert to grayscale (Fixed: cvtColor instead of evtColor)
# Standard for 2025: Ensure the image is successfully loaded before processing
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Simple segmentation (Fixed: cv2.threshold returns a tuple [ret, image])
    ret, segmented = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(8, 4))

    # 4. Display Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    # Convert BGR to RGB for correct colors in Matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    plt.axis('off')

    # 5. Display Segmented Image
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented, cmap="gray")
    plt.axis('off')

    plt.show()  # Fixed from 'pls.show()'
else:
    print("Error: Could not read image. Please check the file path.")