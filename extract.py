import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video frame rate and calculate interval
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(video_fps / fps)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break when video ends

        # Save the frame if it's the desired interval
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
            extracted_count += 1

        frame_count += 1

    # Release video capture and close
    cap.release()
    print(f"Extraction complete. Total frames saved: {extracted_count}")

# Parameters
video_path = "/home/umeshgjh/Documents/HUD Dataset/drive-download-20250116T052555Z-001/L_HUD4.mp4"  # Replace with your video file path
output_dir = "L_HUD4_frames"          # Directory to save frames
fps = 1                               # Frames per second to extract

# Extract frames
extract_frames(video_path, output_dir, fps)
