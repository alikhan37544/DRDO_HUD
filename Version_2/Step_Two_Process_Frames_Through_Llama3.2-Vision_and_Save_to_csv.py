#!/usr/bin/env python3
"""
Improved Script 2: HUD OCR Extraction using Llama3.2-Vision via LangChain's Ollama Integration

Features:
  - Modular code structure with a main() function.
  - Logging for debugging instead of print statements.
  - Retry mechanism if OCR output doesn't match expected format.
  - Optional command-line arguments for input folder and output CSV path.
  - Progress bar via tqdm (if installed) for processing multiple images.
  - Uses Tkinter file dialogs as a fallback if arguments are not provided.

The script processes a folder of images (supported formats: jpg, jpeg, png, bmp),
sends each image to the vision model with a strict OCR prompt to extract three values:
speed, heading, and altitude.
The OCR result must be exactly three comma-separated values (e.g. "350,45,12000").
If not, the script retries a few times before writing a "MISMATCH" row.
Results are saved to a CSV file.
"""

import os
import glob
import csv
import argparse
import logging
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import tkinter as tk
from tkinter import filedialog, messagebox

from langchain_ollama import OllamaLLM  # Updated import from langchain_ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for the vision model
OLLAMA_MODEL = "llama3.2-vision"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ensure this matches your Ollama server config

# Number of retries if the OCR result does not yield exactly 3 values
MAX_RETRIES = 2

# Instantiate the Ollama LLM with num_threads=8 (adjust/remove as needed)
ollama_llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, num_threads=8)

# Define the strict OCR prompt.
# The model is instructed to output exactly three numbers (speed, heading, altitude)
# separated by commas, with no extra text.
STRICT_OCR_PROMPT = (
    "You are a strict OCR for HUD displays. Your task is to extract exactly three numerical values "
    "from the provided HUD image corresponding to speed, heading, and altitude. "
    "Output only these three numbers separated by commas with no additional text, spaces, or punctuation. "
    "For example: '350,45,12000'. If a value cannot be determined, leave its position empty (but maintain the commas).\n\n"
    "Image file: {image_path}"
)

def select_folder_dialog(title="Select Folder Containing Frame Images"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    return folder

def select_csv_save_path_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save CSV File", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def call_vision_model(image_path, prompt_template):
    """
    Calls the vision model using the given prompt.
    Returns the raw response as a string.
    """
    try:
        prompt_with_image = prompt_template.format(image_path=image_path)
        logging.info("Sending prompt for %s:\n%s", image_path, prompt_with_image)
        response = ollama_llm.invoke(prompt_with_image)
        raw_response = response.strip()
        logging.info("Raw model response for %s:\n%s", image_path, raw_response)
        return raw_response
    except Exception as e:
        logging.error("Error processing %s: %s", image_path, e)
        return None

def process_image(image_path):
    """
    Process a single image using the OCR prompt.
    Retries up to MAX_RETRIES times if the OCR output does not return exactly three comma-separated values.
    Returns a list of three values or ["MISMATCH", "MISMATCH", "MISMATCH"] on failure.
    """
    for attempt in range(MAX_RETRIES + 1):
        response = call_vision_model(image_path, STRICT_OCR_PROMPT)
        if response is None:
            continue
        values = [v.strip() for v in response.split(",")]
        if len(values) == 3:
            return values
        else:
            logging.warning("Attempt %d: Parameter count mismatch for %s. Expected 3 values, got %d. Response: %s",
                            attempt + 1, image_path, len(values), response)
            time.sleep(1)  # Delay before retrying
    return ["MISMATCH", "MISMATCH", "MISMATCH"]

def process_frames(input_folder, csv_save_path):
    # Gather image files (supports jpg, jpeg, png, bmp)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files.sort()
    
    if not image_files:
        logging.error("No image files found in the selected folder: %s", input_folder)
        return

    headers = ["frame", "image_file", "speed", "heading", "altitude"]
    with open(csv_save_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        
        # Use a progress bar if tqdm is available
        iterator = tqdm(image_files, desc="Processing images") if tqdm else image_files
        for image_path in iterator:
            values = process_image(image_path)
            row = [os.path.basename(image_path), image_path] + values
            csvwriter.writerow(row)
            logging.info("Processed %s => %s", image_path, values)
    
    logging.info("CSV file successfully created at: %s", csv_save_path)

def main():
    parser = argparse.ArgumentParser(description="HUD OCR Extraction using Llama3.2-Vision")
    parser.add_argument("--input_folder", help="Path to folder containing frame images")
    parser.add_argument("--output_csv", help="Path to output CSV file")
    args = parser.parse_args()
    
    if args.input_folder:
        input_folder = args.input_folder
    else:
        input_folder = select_folder_dialog("Select Folder Containing Frame Images")
        if not input_folder:
            messagebox.showerror("Error", "No folder selected.")
            return
    
    if args.output_csv:
        csv_save_path = args.output_csv
    else:
        csv_save_path = select_csv_save_path_dialog()
        if not csv_save_path:
            messagebox.showerror("Error", "No CSV file path selected.")
            return

    process_frames(input_folder, csv_save_path)

if __name__ == "__main__":
    main()
