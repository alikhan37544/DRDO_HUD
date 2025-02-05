import cv2
import easyocr
import pandas as pd

# Load the video
video_path = "/home/understressengineer/programming/DRDO_HUD/VID-20250127-WA0006.mp4"  # Update with the actual video path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 0
data = []

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Initialize EasyOCR reader with an allowlist of digits
reader = easyocr.Reader(['en'])

# Define bounding box coordinates for the "Heading" HUD element (adjust as needed)
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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Extract heading
    heading = extract_heading(frame)

    # Log extracted heading
    print(f"Frame {frame_count}: Heading: {heading}")

    # Add frame number and timestamp to data
    data.append({
        "Frame": frame_count,
        "Timestamp": frame_count / fps,
        "Heading": heading
    })

    # Overlay extracted heading on frame
    overlay_text = f"Heading: {heading}"
    cv2.putText(frame, overlay_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    frame_count += 1

    # Display for debugging (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save extracted data to CSV
if data:
    df = pd.DataFrame(data)
    df.to_csv("extracted_data.csv", index=False)
    print("CSV file saved successfully.")
else:
    print("No data extracted, CSV not saved.")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete.")
