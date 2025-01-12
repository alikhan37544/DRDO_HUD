import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

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

def detect_and_overlay_horizon(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the input video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        horizon_y = height // 2

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:
                    horizon_y = (y1 + y2) // 2
                    break

        overlay = frame.copy()
        cv2.line(overlay, (0, horizon_y), (width, horizon_y), (0, 255, 0), 2)
        cv2.rectangle(overlay, (0, 0), (width, horizon_y), (255, 0, 0), -1)
        cv2.rectangle(overlay, (0, horizon_y), (width, height), (0, 0, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        out.write(frame)

    cap.release()
    out.release()
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

    detect_and_overlay_horizon(input_path, output_path)

if __name__ == "__main__":
    main()
