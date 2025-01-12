import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

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

def detect_horizon_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
    if lines is not None:
        # Find the most horizontal line
        horizon_line = max(lines, key=lambda line: abs(line[0][2] - line[0][0]))
        x1, y1, x2, y2 = horizon_line[0]
        return x1, y1, x2, y2
    else:
        return None

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        horizon_coords = detect_horizon_line(frame)
        if horizon_coords:
            x1, y1, x2, y2 = horizon_coords
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Horizon Detection', frame)
        out.write(frame)

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
