#!/usr/bin/env python3
"""
Script 2: HUD OCR Extraction using Llama3.2-Vision via LangChain's Ollama Integration (Simplified)

This script performs the following:
  1. Lets the user select a folder containing frame images.
  2. For each image, sends a prompt to the vision model instructing it to act as a strict OCR.
     The prompt extracts exactly three parameters: speed, heading, and altitude.
  3. The model must output exactly three comma-separated numerical values with no extra text.
  4. The script prints the raw model response (for debugging) and saves the results into a CSV file 
     with columns: frame, image_file, speed, heading, altitude.

Notes:
  - This script uses the updated LangChain Ollama integration from `langchain_ollama` (install via `pip install -U langchain-ollama`).
  - The LLM is instantiated with num_threads=8 for faster processing on multi-thread systems (remove or adjust if running on a different system).
"""

import os
import glob
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

from langchain_ollama import OllamaLLM  # Updated import from langchain_ollama

# Configuration for the vision model
OLLAMA_MODEL = "llama3.2-vision"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ensure this matches your Ollama server config

# Instantiate the Ollama LLM with num_threads=8 for faster processing.
# (Remove or adjust num_threads if running on a different system)
ollama_llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, num_threads=8)

# Define a strict OCR prompt.
# The prompt instructs the model to extract exactly three numerical values (speed, heading, altitude)
# with no extra text. For example: "350,45,12000"
STRICT_OCR_PROMPT = (
    "You are a strict OCR for HUD displays. Your task is to extract exactly three numerical values "
    "from the provided HUD image corresponding to speed, heading, and altitude. "
    "Output only these three numbers separated by commas with no additional text, spaces, or punctuation. "
    "For example: '350,45,12000'. If a value cannot be determined, leave its position empty (but maintain the commas).\n\n"
    "Image file: {image_path}"
)

def select_folder(title="Select Folder Containing Frame Images"):
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

def call_vision_model(image_path, prompt_template):
    """
    Calls the vision model using the given prompt template.
    The image_path is inserted into the prompt so that the model will load the image.
    Prints the prompt and raw response for debugging.
    """
    try:
        prompt_with_image = prompt_template.format(image_path=image_path)
        print(f"Sending prompt for {image_path}:\n{prompt_with_image}\n")
        response = ollama_llm.invoke(prompt_with_image)
        raw_response = response.strip()
        print(f"Raw model response for {image_path}:\n{raw_response}\n")
        return raw_response
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_frames(folder, csv_save_path):
    # Gather image files (supports jpg, jpeg, png, bmp)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    image_files.sort()  # sort alphabetically

    if not image_files:
        messagebox.showerror("Error", "No image files found in the selected folder.")
        return

    headers = ["frame", "image_file", "speed", "heading", "altitude"]
    with open(csv_save_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        for image_path in image_files:
            response = call_vision_model(image_path, STRICT_OCR_PROMPT)
            if response is None:
                values = ["ERROR", "ERROR", "ERROR"]
            else:
                # Expect exactly three values separated by commas.
                values = [v.strip() for v in response.split(",")]
                if len(values) != 3:
                    print(f"Warning: Parameter count mismatch for {image_path}. Expected 3 values.")
                    values = ["MISMATCH", "MISMATCH", "MISMATCH"]
            row = [os.path.basename(image_path), image_path] + values
            csvwriter.writerow(row)
            print(f"Processed {image_path} => {values}\n")

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
