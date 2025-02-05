import cv2
import pytesseract
import pandas as pd
import re

# Path to Tesseract OCR (modify based on your OS)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Modify this path

# Load video
video_path = "/home/understressengineer/programming/DRDO_HUD/VID-20250127-WA0003.mp4"  # Replace with your actual video path
cap = cv2.VideoCapture(video_path)

# Get video FPS for timestamp calculation
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
data = []

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

def extract_data(text):
    """ Extract relevant fields from OCR text using regex """
    heading = re.search(r'HEADING\s*([\d.]+)', text)
    speed = re.search(r'SPEED\s*([\d.]+)', text)
    altitude = re.search(r'ALT\s*([\d.]+)', text)
    time = re.search(r'TIME\s*([\d.]+)', text)

    return {
        "time": float(time.group(1)) if time else None,
        "heading": float(heading.group(1)) if heading else None,
        "speed": float(speed.group(1)) if speed else None,
        "altitude": float(altitude.group(1)) if altitude else None
    }

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for better OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    text = pytesseract.image_to_string(gray, config="--psm 6")

    # Extract data
    extracted_data = extract_data(text)
    extracted_data["frame"] = frame_count
    extracted_data["timestamp"] = frame_count / fps  # Calculate timestamp
    data.append(extracted_data)

    # Overlay extracted text on frame
    overlay_text = f"Heading: {extracted_data['heading']} | Speed: {extracted_data['speed']} | Alt: {extracted_data['altitude']} | Time: {extracted_data['time']}"
    cv2.putText(frame, overlay_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    frame_count += 1

    # Display for debugging (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save data to CSV
df = pd.DataFrame(data)
df.to_csv("extracted_data.csv", index=False)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
