#!/usr/bin/env python3
"""
Template-Based HUD OCR Extraction using Llama3.2-Vision via LangChain's Ollama Integration

This script does the following:
  1. Lets the user select a folder containing frame images.
  2. For each image, sends a prompt to the vision model instructing it to act as an OCR engine.
     The prompt instructs the model to extract exactly three data points (speed, heading, altitude)
     and output them strictly in the following format:
     
         *** Speed = <speed_value>
         &&& Heading = <heading_value>
         ^^^ Altitude = <altitude_value>
     
     If a value cannot be determined, leave it empty (but maintain the "=").
  3. The script then parses the output according to this template and writes the results into a CSV file.
  
Notes:
  - This script uses the updated LangChain Ollama integration from `langchain_ollama`.
  - The LLM is instantiated with num_threads=8 (adjust if needed).
  - A retry mechanism is provided if the output format isn’t as expected.
  - Logging is used for debugging.
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

from langchain_ollama import OllamaLLM  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for the vision model
OLLAMA_MODEL = "llama3.2-vision"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Adjust to match your Ollama server config

# Maximum number of retries if the output format is not as expected
MAX_RETRIES = 2

# Instantiate the Ollama LLM with num_threads=8 (adjust or remove as needed)
ollama_llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, num_threads=8)

# Define the template OCR prompt.
TEMPLATE_OCR_PROMPT = (
    "You are an OCR specialized for fighter jet HUD displays. I will provide you an image file. "
    "Your task is to extract exactly three data points from the HUD: Speed, Heading, and Altitude. "
    "Output them in the following strict format (with no additional text or commentary):\n\n"
    "*** Speed = <speed_value>\n"
    "&&& Heading = <heading_value>\n"
    "^^^ Altitude = <altitude_value>\n\n"
    "Replace <speed_value>, <heading_value>, and <altitude_value> with the numerical values extracted from the image. "
    "If a value cannot be determined, leave it empty after the '=' sign.\n\n"
    "Example output:\n"
    "*** Speed = 350\n"
    "&&& Heading = 45\n"
    "^^^ Altitude = 12000\n\n"
    "Do not output any additional text. Output only the three lines exactly as specified.\n\n"
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
    Calls the vision model using the given prompt template.
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

def parse_template_output(raw_output):
    """
    Parses the output from the model which should be exactly three lines in the template format:
      *** Speed = <speed_value>
      &&& Heading = <heading_value>
      ^^^ Altitude = <altitude_value>
    Returns a list of three values [speed, heading, altitude].
    If the format does not match exactly, returns None.
    """
    lines = raw_output.splitlines()
    if len(lines) != 3:
        return None
    try:
        # Each line should split exactly on '=' into two parts.
        speed_line, heading_line, altitude_line = lines
        if not speed_line.startswith("*** Speed ="):
            return None
        if not heading_line.startswith("&&& Heading ="):
            return None
        if not altitude_line.startswith("^^^ Altitude ="):
            return None
        
        # Extract values after '=' and strip any spaces.
        speed = speed_line.split("=", 1)[1].strip()
        heading = heading_line.split("=", 1)[1].strip()
        altitude = altitude_line.split("=", 1)[1].strip()
        return [speed, heading, altitude]
    except Exception as e:
        logging.error("Error parsing template output: %s", e)
        return None

def process_image(image_path):
    """
    Process a single image using the template OCR prompt.
    Retries up to MAX_RETRIES times if the output is not in the expected format.
    Returns a list of three values (speed, heading, altitude) or ["MISMATCH", "MISMATCH", "MISMATCH"] on failure.
    """
    for attempt in range(MAX_RETRIES + 1):
        raw_response = call_vision_model(image_path, TEMPLATE_OCR_PROMPT)
        if raw_response is None:
            continue
        parsed = parse_template_output(raw_response)
        if parsed is not None:
            return parsed
        else:
            logging.warning("Attempt %d: Template output mismatch for %s. Response:\n%s",
                            attempt + 1, image_path, raw_response)
            time.sleep(1)  # Wait before retrying
    return ["MISMATCH", "MISMATCH", "MISMATCH"]

def process_frames(input_folder, csv_save_path):
    # Gather image files (supports jpg, jpeg, png, bmp)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files.sort()

    if not image_files:
        logging.error("No image files found in the folder: %s", input_folder)
        return

    headers = ["frame", "image_file", "speed", "heading", "altitude"]
    with open(csv_save_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        iterator = tqdm(image_files, desc="Processing images") if tqdm else image_files
        for image_path in iterator:
            values = process_image(image_path)
            row = [os.path.basename(image_path), image_path] + values
            csvwriter.writerow(row)
            logging.info("Processed %s => %s", image_path, values)

    logging.info("CSV file successfully created at: %s", csv_save_path)

def main():
    parser = argparse.ArgumentParser(description="HUD Template OCR Extraction using Llama3.2-Vision")
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
