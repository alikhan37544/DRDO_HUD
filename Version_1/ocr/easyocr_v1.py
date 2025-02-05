import cv2
import easyocr
import pandas as pd

# Load the video
video_path = "/home/understressengineer/programming/DRDO_HUD/VID-20250127-WA0003.mp4"  # Update with actual video path
cap = cv2.VideoCapture(video_path)

# Get video FPS for timestamp calculation
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
data = []

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define bounding box coordinates for each HUD element (adjust as needed)
regions = {
    "Heading": (60, 35, 200, 75),  # (x1, y1, x2, y2)
    "Speed": (60, 130, 200, 170),
    "Altitude": (390, 130, 530, 170),
    "Time": (390, 35, 530, 75),
}

def extract_text_from_region(image, bbox):
    """ Crop the region and extract text using EasyOCR """
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]  # Crop region

    # Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR for text detection
    result = reader.readtext(roi_gray, detail=0)  # Get only text, no bounding boxes

    return result[0] if result else "N/A"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract data for each field
    extracted_data = {key: extract_text_from_region(frame, bbox) for key, bbox in regions.items()}

    # Add frame number and timestamp
    extracted_data["Frame"] = frame_count
    extracted_data["Timestamp"] = frame_count / fps
    data.append(extracted_data)

    # Overlay extracted text on frame
    overlay_text = f"Heading: {extracted_data['Heading']} | Speed: {extracted_data['Speed']} | Alt: {extracted_data['Altitude']} | Time: {extracted_data['Time']}"
    cv2.putText(frame, overlay_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    frame_count += 1

    # Display for debugging (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save extracted data to CSV
df = pd.DataFrame(data)
df.to_csv("extracted_data.csv", index=False)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
