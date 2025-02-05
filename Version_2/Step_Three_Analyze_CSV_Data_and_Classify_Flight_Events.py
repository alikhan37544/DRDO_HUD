#!/usr/bin/env python3
"""
Script 2: HUD Data Extraction using Llama3.2-Vision via Ollama
- Allows the user to select a folder containing frame images.
- Uses the first frame to ask the vision model what parameters (HUD details) are present.
- Generates a CSV file with headers based on the parameters.
- Processes each image with a strong prompt that extracts only the values (like an OCR).
- Saves the results into the CSV file.
"""

import os
import glob
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from PIL import Image

# Configuration
OLLAMA_HOST = "http://127.0.0.1:11434"  # Ollama API host

# Define strong prompts
FIRST_PROMPT = (
    "Analyze the provided HUD image carefully. List all the key parameters "
    "and indicators you can observe (such as speed, altitude, heading, etc.) "
    "that are relevant for flight data analysis. Provide the list as comma-separated values."
)

SECOND_PROMPT_TEMPLATE = (
    "Given the following HUD parameters: {params}. "
    "For the provided image, extract only the corresponding values for each parameter. "
    "Return the values in the same comma-separated order with no additional text or commentary."
)

def select_folder(title="Select Folder Containing Frames"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    return folder

def select_csv_save_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save CSV File", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def call_vision_model(image_path, prompt):
    """
    Sends the image along with the prompt to the vision model API.
    Assumes that the API accepts multipart/form-data with 'file' and 'prompt' fields.
    """
    with open(image_path, "rb") as img_file:
        files = {"file": img_file}
        data = {"prompt": prompt}
        try:
            response = requests.post(OLLAMA_HOST, data=data, files=files)
            response.raise_for_status()
            # Assuming the API returns a JSON with a 'result' field
            result = response.json().get("result", "")
            return result.strip()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def parse_parameters(param_str):
    """
    Parse the comma separated list of parameters.
    The vision model is expected to return something like: "Speed, Altitude, Heading, ..."
    """
    params = [p.strip() for p in param_str.split(",") if p.strip()]
    return params

def process_frames(folder, csv_save_path):
    # Get list of image files (common image extensions)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    image_files.sort()  # sort by name

    if not image_files:
        messagebox.showerror("Error", "No image files found in the folder.")
        return

    # Step 1: Use the first image to determine the HUD parameters
    first_image = image_files[0]
    print(f"Using first image for parameter extraction: {first_image}")
    param_response = call_vision_model(first_image, FIRST_PROMPT)
    if not param_response:
        print("Could not extract parameters from the first image.")
        return

    print("Vision model parameters response:", param_response)
    parameters = parse_parameters(param_response)
    if not parameters:
        print("No valid parameters parsed from the vision model response.")
        return

    # Create CSV file with headers: a frame identifier and the parameters.
    headers = ["frame", "image_file"] + parameters
    with open(csv_save_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        # Step 2: Process each frame image
        for image_path in image_files:
            # Build the second prompt with the parameters list
            second_prompt = SECOND_PROMPT_TEMPLATE.format(params=", ".join(parameters))
            result = call_vision_model(image_path, second_prompt)
            if result is None:
                row = [os.path.basename(image_path), image_path] + ["ERROR"] * len(parameters)
            else:
                # Expecting a comma-separated string of values
                values = [v.strip() for v in result.split(",")]
                # If the number of returned values does not match parameters, mark error
                if len(values) != len(parameters):
                    print(f"Warning: Parameter count mismatch for {image_path}. Expected {len(parameters)} values.")
                    values = ["MISMATCH"] * len(parameters)
                row = [os.path.basename(image_path), image_path] + values

            csvwriter.writerow(row)
            print(f"Processed {image_path}")

    print(f"CSV file created at {csv_save_path}")

if __name__ == "__main__":
    frames_folder = select_folder("Select Folder Containing Frame Images")
    if not frames_folder:
        messagebox.showerror("Error", "No folder selected.")
        exit(1)

    csv_path = select_csv_save_path()
    if not csv_path:
        messagebox.showerror("Error", "No CSV file path selected.")
        exit(1)

    process_frames(frames_folder, csv_path)
