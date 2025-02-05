#!/usr/bin/env python3
"""
Script 1: Video to Frames Converter with GUI
- Allows the user to select a video file.
- Lets the user choose an output folder and name it.
- The user can specify a time interval (in seconds) to extract frames.
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def select_video_file():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.flv"), ("All Files", "*.*")]
    )
    return video_path

def select_output_directory():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Output Directory")
    return folder

def get_interval():
    root = tk.Tk()
    root.withdraw()
    interval = simpledialog.askfloat("Frame Interval", "Enter frame interval in seconds:", minvalue=0.1)
    return interval

def video_to_frames(video_path, output_folder, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    print(f"Video FPS: {fps}, extracting every {frame_interval} frames.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
            print(f"Saved {frame_filename}")

        frame_num += 1

    cap.release()
    print(f"Done. Total frames saved: {saved_frames}")

if __name__ == "__main__":
    video_file = select_video_file()
    if not video_file:
        messagebox.showerror("Error", "No video file selected.")
        exit(1)

    output_dir = select_output_directory()
    if not output_dir:
        messagebox.showerror("Error", "No output folder selected.")
        exit(1)

    interval_sec = get_interval()
    if not interval_sec:
        messagebox.showerror("Error", "Invalid interval entered.")
        exit(1)

    # Create a dedicated folder within the selected directory if desired
    folder_name = os.path.splitext(os.path.basename(video_file))[0] + "_frames"
    full_output_path = os.path.join(output_dir, folder_name)
    os.makedirs(full_output_path, exist_ok=True)

    video_to_frames(video_file, full_output_path, interval_sec)
