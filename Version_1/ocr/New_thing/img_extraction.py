import cv2
import numpy as np
import pytesseract

# If Tesseract is not auto-detected on Arch, specify the full path:
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def extract_text_from_green(image):
    """
    1. Convert ROI to HSV
    2. Threshold on the "green" range (you may need to tweak values)
    3. Convert mask to BGR so Tesseract can read it
    4. Use Tesseract to extract text
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Approximate green range – adjust as needed
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Try a Tesseract config that treats the region as a single line
    config = "--psm 7"
    text = pytesseract.image_to_string(mask_bgr, config=config)
    return text.strip()

def safe_float(value_str):
    """Convert the recognized text to float, or None if it fails."""
    try:
        return float(value_str)
    except ValueError:
        return None

def main():
    # 1. Load your HUD image
    image_path = "hud_frame.png"  # <-- Replace with your actual screenshot
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image from", image_path)
        return

    # 2. Define bounding boxes for each numeric field
    #    NOTE: (x1, y1, x2, y2) = left, top, right, bottom
    #    You must adjust these based on your image’s actual coordinates!
    heading_roi = (90,  50, 180,  90)   # Approx for "136.4"
    speed_roi   = (90, 180, 180, 220)   # Approx for "289.8"
    alt_roi     = (365, 180, 460, 220)  # Approx for "9.376"

    # 3. Extract each ROI from the main image
    #    Remember OpenCV slices are [y1:y2, x1:x2]
    (hx1, hy1, hx2, hy2) = heading_roi
    heading_crop = frame[hy1:hy2, hx1:hx2]
    (sx1, sy1, sx2, sy2) = speed_roi
    speed_crop = frame[sy1:sy2, sx1:sx2]
    (ax1, ay1, ax2, ay2) = alt_roi
    alt_crop = frame[ay1:ay2, ax1:ax2]

    # 4. Extract text by color masking and Tesseract
    heading_text = extract_text_from_green(heading_crop)
    speed_text   = extract_text_from_green(speed_crop)
    alt_text     = extract_text_from_green(alt_crop)

    # 5. Convert recognized strings to floats
    heading_val = safe_float(heading_text)
    speed_val   = safe_float(speed_text)
    alt_val     = safe_float(alt_text)

    # 6. Print out the results
    print("Heading:", heading_val)
    print("Speed:  ", speed_val)
    print("Altitude:", alt_val)

if __name__ == "__main__":
    main()
