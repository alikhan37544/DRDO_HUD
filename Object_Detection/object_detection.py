import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk

def select_input_file():
    input_file = filedialog.askopenfilename(
        title="Select Input Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    return input_file

def select_output_file():
    output_file = filedialog.asksaveasfilename(
        title="Save Output Video As",
        defaultextension=".mp4",
        filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")]
    )
    return output_file

def draw_hud_and_detections(frame, detections):
    # HUD elements can be added here (e.g., artificial horizon, altitude, etc.)
    # For simplicity, this example focuses on object detection overlays.

    # Draw detection boxes for detected aircraft
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['box'])
        confidence = detection['confidence']
        label = detection['label']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the input video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load the YOLOv11 model
    model = YOLO('yolov11.pt')  # Ensure 'yolov11.pt' is the correct path to your model weights

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        detections = []
        for result in results:
            for box, score, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if cls == 'airplane' and score > 0.5:  # Adjust class name and confidence threshold as needed
                    detections.append({
                        'box': box,
                        'confidence': score,
                        'label': 'Aircraft'
                    })

        # Draw HUD and detection overlays
        frame = draw_hud_and_detections(frame, detections)
        out.write(frame)

        # Display the frame with HUD and detections in real-time
        cv2.imshow('HUD with Aircraft Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", "Processing complete. Output saved.")

def main():
    root = tk.Tk()
    root.withdraw()

    input_path = select_input_file()
    if not input_path:
        messagebox.showwarning("Input Required", "No input video selected.")
        return

    output_path = select_output_file()
    if not output_path:
        messagebox.showwarning("Output Required", "No output file specified.")
        return

    process_video(input_path, output_path)

if __name__ == "__main__":
    main()
