#!/usr/bin/env python3
"""
Script 2: HUD Data Extraction using Llama3.2-Vision via LangChain’s Ollama Integration

This script:
  1. Lets the user select a folder containing frame images.
  2. Uses the first frame to ask the vision model what HUD parameters are present.
  3. Creates a CSV file with headers based on those parameters.
  4. For every frame, sends a second prompt (with the image embedded as Base64)
     to extract only the numeric/word values corresponding to those parameters.
  5. Saves the extracted data into the CSV file.

Dependencies:
  - langchain (pip install langchain)
  - tkinter (usually comes with Python)
  - base64, glob, csv, os (standard library)
  
Note: This script assumes that your “llama3.2-vision” model (served via Ollama) 
accepts image input when provided in Markdown format, e.g.:

    ![image](data:image/jpeg;base64,<BASE64_STRING>)

Adjust the prompt or integration details if your setup requires a different format.
"""

import os
import glob
import csv
import base64
import tkinter as tk
from tkinter import filedialog, messagebox

from langchain.llms import Ollama

# Configuration for the vision model
OLLAMA_MODEL = "llama3.2-vision"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ensure this matches your ollama serve config

# Instantiate the Ollama LLM via LangChain
ollama_llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# Define the strong prompts
FIRST_PROMPT = (
    "Analyze the provided HUD image carefully. List all the key parameters "
    "and indicators you can observe (for example: speed, altitude, heading, etc.) "
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

def encode_image_to_base64(image_path):
    """Load an image file and encode it as a base64 string."""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded_string

def call_vision_model(image_path, prompt):
    """
    Uses LangChain’s Ollama LLM to send a prompt with the image embedded as a Base64-encoded markdown image.
    """
    try:
        encoded_image = encode_image_to_base64(image_path)
        # Embed the image using Markdown syntax
        prompt_with_image = f"{prompt}\n\n![image](data:image/jpeg;base64,{encoded_image})"
        response = ollama_llm(prompt_with_image)
        return response.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def parse_parameters(param_str):
    """
    Parses the comma-separated list of parameters returned by the vision model.
    """
    params = [p.strip() for p in param_str.split(",") if p.strip()]
    return params

def process_frames(folder, csv_save_path):
    # Gather image files with common image extensions
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    image_files.sort()  # sort alphabetically

    if not image_files:
        messagebox.showerror("Error", "No image files found in the selected folder.")
        return

    # Step 1: Use the first image to extract the HUD parameters
    first_image = image_files[0]
    print(f"Using first image for parameter extraction: {first_image}")
    param_response = call_vision_model(first_image, FIRST_PROMPT)
    if not param_response:
        print("Could not extract parameters from the first image.")
        return

    print("Vision model parameter response:", param_response)
    parameters = parse_parameters(param_response)
    if not parameters:
        print("No valid parameters parsed from the vision model response.")
        return

    # Create CSV file headers (frame identifier, image file, plus the extracted parameters)
    headers = ["frame", "image_file"] + parameters
    with open(csv_save_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        # Step 2: Process each frame image to extract the corresponding values
        for image_path in image_files:
            second_prompt = SECOND_PROMPT_TEMPLATE.format(params=", ".join(parameters))
            result = call_vision_model(image_path, second_prompt)
            if result is None:
                row = [os.path.basename(image_path), image_path] + ["ERROR"] * len(parameters)
            else:
                # Expect a comma-separated string of values
                values = [v.strip() for v in result.split(",")]
                if len(values) != len(parameters):
                    print(f"Warning: Parameter count mismatch for {image_path}. Expected {len(parameters)} values.")
                    values = ["MISMATCH"] * len(parameters)
                row = [os.path.basename(image_path), image_path] + values

            csvwriter.writerow(row)
            print(f"Processed {image_path}")

    print(f"CSV file successfully created at: {csv_save_path}")

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
