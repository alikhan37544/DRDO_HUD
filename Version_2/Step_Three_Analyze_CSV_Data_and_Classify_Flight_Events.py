#!/usr/bin/env python3
"""
Flight Event Classifier from HUD CSV Data

This script:
  - Prompts the user to select the CSV file generated in Step 2.
  - Reads the CSV (which contains columns such as "frame", "image_file", and various HUD parameters).
  - Compares consecutive rows to detect significant changes based on preset thresholds.
  - Applies simple rules to classify events (e.g. if speed increases significantly while altitude decreases, label as "Pull Up").
  - Writes a new CSV file (or overwrites if desired) with an additional "event" column that summarizes the detected events.

Note:
  - Thresholds for speed, altitude, and heading are defined in the script – adjust these as necessary.
  - The script uses pandas for CSV processing and tkinter for file selection dialogs.
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# Define threshold values (adjust these based on your data and units)
THRESHOLDS = {
    "speed": 50,       # e.g., a change in speed (units as per your CSV data)
    "altitude": 1000,  # e.g., a change in altitude (units as per your CSV data)
    "heading": 10      # e.g., a change in heading (degrees)
}

def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV File with HUD Data", filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def select_output_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Save Classified Events CSV", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def classify_events(df):
    """
    For each row (after the first), compare with the previous row and
    classify events based on significant changes.
    
    Returns the DataFrame with an additional "event" column.
    """
    # Assume the first two columns are "frame" and "image_file".
    # The remaining columns are the extracted HUD parameters.
    param_cols = df.columns[2:]
    
    # Convert parameter columns to numeric (coerce errors to NaN)
    for col in param_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Try to detect which column corresponds to which parameter using a case-insensitive search.
    col_map = {}
    for col in param_cols:
        clower = col.lower()
        if "speed" in clower:
            col_map["speed"] = col
        elif "altitude" in clower:
            col_map["altitude"] = col
        elif "heading" in clower:
            col_map["heading"] = col

    # Prepare a list to store event descriptions for each row.
    events = ["N/A"]  # First row has no previous data to compare.

    # Iterate over rows (starting from row 1) and compare with previous row.
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        event_desc = []
        
        # Check speed change if available.
        if "speed" in col_map:
            prev_speed = prev[col_map["speed"]]
            curr_speed = curr[col_map["speed"]]
            if pd.notna(prev_speed) and pd.notna(curr_speed):
                diff_speed = curr_speed - prev_speed
                if abs(diff_speed) >= THRESHOLDS["speed"]:
                    if diff_speed > 0:
                        event_desc.append(f"Speed Increase ({prev_speed}->{curr_speed})")
                    else:
                        event_desc.append(f"Speed Decrease ({prev_speed}->{curr_speed})")
        
        # Check altitude change if available.
        if "altitude" in col_map:
            prev_alt = prev[col_map["altitude"]]
            curr_alt = curr[col_map["altitude"]]
            if pd.notna(prev_alt) and pd.notna(curr_alt):
                diff_alt = curr_alt - prev_alt
                if abs(diff_alt) >= THRESHOLDS["altitude"]:
                    if diff_alt > 0:
                        event_desc.append(f"Altitude Increase ({prev_alt}->{curr_alt})")
                    else:
                        event_desc.append(f"Altitude Decrease ({prev_alt}->{curr_alt})")
        
        # Check heading change if available.
        if "heading" in col_map:
            prev_head = prev[col_map["heading"]]
            curr_head = curr[col_map["heading"]]
            if pd.notna(prev_head) and pd.notna(curr_head):
                diff_head = curr_head - prev_head
                if abs(diff_head) >= THRESHOLDS["heading"]:
                    if diff_head > 0:
                        event_desc.append(f"Heading Increase ({prev_head}->{curr_head})")
                    else:
                        event_desc.append(f"Heading Decrease ({prev_head}->{curr_head})")
        
        # Combined rule: If speed increases significantly and altitude decreases significantly,
        # classify it as a "Pull Up" maneuver.
        if "speed" in col_map and "altitude" in col_map:
            if pd.notna(prev[col_map["speed"]]) and pd.notna(curr[col_map["speed"]]) \
               and pd.notna(prev[col_map["altitude"]]) and pd.notna(curr[col_map["altitude"]]):
                if (curr[col_map["speed"]] - prev[col_map["speed"]]) > THRESHOLDS["speed"] \
                   and (curr[col_map["altitude"]] - prev[col_map["altitude"]]) < -THRESHOLDS["altitude"]:
                    event_desc.append("Pull Up")
        
        # If no significant change is detected, mark as "No Significant Change"
        if not event_desc:
            event_desc = ["No Significant Change"]
        
        events.append("; ".join(event_desc))
    
    df["event"] = events
    return df

def main():
    csv_path = select_csv_file()
    if not csv_path:
        messagebox.showerror("Error", "No CSV file selected.")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error reading CSV: {e}")
        return
    
    classified_df = classify_events(df)
    
    output_path = select_output_file()
    if not output_path:
        messagebox.showerror("Error", "No output file selected.")
        return
    try:
        classified_df.to_csv(output_path, index=False)
        print(f"Flight events classification saved to {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Error writing CSV: {e}")

if __name__ == "__main__":
    main()
