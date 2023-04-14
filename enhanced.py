import cv2
import os

# Load the image
img_name = 'image4.png'
img = cv2.imread(img_name)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
eq_gray = cv2.equalizeHist(gray)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_gray = clahe.apply(gray)

# Save the enhanced image in the same directory
name, ext = os.path.splitext(img_name)
enhanced_name = f"{name}_enhanced.png"
cv2.imwrite(enhanced_name, clahe_gray)

# Display the original, histogram equalized, and CLAHE images
cv2.imshow('Original', img)
cv2.imshow('Histogram Equalized', eq_gray)
cv2.imshow('CLAHE', clahe_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
