import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load the image
img = cv2.imread("image4.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to pre-process the image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Pass the image to Tesseract OCR
text = pytesseract.image_to_string(thresh)

# Write the extracted text to a file
with open("output.txt", "w") as f:
    f.write(text)
