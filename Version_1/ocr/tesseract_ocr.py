import cv2
import pytesseract
import csv
import re

# If tesseract is not auto-detected, uncomment and specify full path:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

def parse_hud_text(hud_text):
    """
    Attempts to parse heading, speed, and altitude from the HUD text using regular expressions.
    Adapt this to match the actual text in your video frames.
    """
    heading_pattern = re.compile(r"Heading[:=]\s*(\d+)", re.IGNORECASE)
    speed_pattern   = re.compile(r"Speed[:=]\s*(\d+)", re.IGNORECASE)
    alt_pattern     = re.compile(r"Altitude[:=]\s*(\d+)", re.IGNORECASE)

    heading  = None
    speed    = None
    altitude = None

    heading_match = heading_pattern.search(hud_text)
    speed_match   = speed_pattern.search(hud_text)
    alt_match     = alt_pattern.search(hud_text)

    if heading_match:
        heading = heading_match.group(1)
    if speed_match:
        speed = speed_match.group(1)
    if alt_match:
        altitude = alt_match.group(1)

    return {
        "heading": heading,
        "speed": speed,
        "altitude": altitude
    }

def main():
    input_video_path = "/home/understressengineer/programming/DRDO_HUD/VID-20250127-WA0004.mp4"   # <-- Change this to your video file
    output_video_path = "output.avi"
    output_csv_path = "extracted_data.csv"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # or *"MJPG"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Prepare CSV file
    csv_file = open(output_csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["frame_index", "heading", "speed", "altitude"])
    csv_writer.writeheader()

    # Adjust ROI to where the HUD text actually is.
    # Example: top-left corner
    roi_x1, roi_y1 = 10, 10
    roi_x2, roi_y2 = 300, 100

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract region of interest
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Convert ROI to grayscale for better OCR
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Additional preprocessing can help, e.g. thresholding:
        # gray_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY)[1]

        # OCR
        hud_text = pytesseract.image_to_string(gray_roi)

        # Parse text
        params = parse_hud_text(hud_text)
        heading  = params["heading"] if params["heading"] else "N/A"
        speed    = params["speed"] if params["speed"] else "N/A"
        altitude = params["altitude"] if params["altitude"] else "N/A"

        # Write to CSV
        csv_writer.writerow({
            "frame_index": frame_index,
            "heading": heading,
            "speed": speed,
            "altitude": altitude
        })

        # Overlay the extracted data on the frame
        overlay_text = f"Heading: {heading}, Speed: {speed}, Altitude: {altitude}"
        cv2.putText(
            frame,
            overlay_text,
            (10, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Write the frame
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    csv_file.close()
    print("Processing complete. Results saved to:")
    print(" - Video:", output_video_path)
    print(" - CSV:  ", output_csv_path)

if __name__ == "__main__":
    main()
