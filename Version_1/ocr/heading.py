import cv2
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define bounding box for "Heading" (adjust coordinates as needed)
heading_bbox = (55, 170, 140, 210)  # (x1, y1, x2, y2)

def preprocess_image(image):
    """Apply preprocessing steps to enhance image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary

def extract_heading(frame):
    """Extract the heading value from the frame."""
    x1, y1, x2, y2 = heading_bbox
    roi = frame[y1:y2, x1:x2]
    preprocessed_roi = preprocess_image(roi)
    result = reader.readtext(
        preprocessed_roi, allowlist='0123456789', detail=0
    )
    return result[0] if result else "N/A"

# Example usage with a single frame (replace with your frame extraction logic)
frame = cv2.imread('path_to_your_frame.jpg')  # Load a frame for testing
heading = extract_heading(frame)
print(f"Extracted Heading: {heading}")
