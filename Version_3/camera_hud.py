import pygame
import cv2
import numpy as np
import math
import time
import os
import csv
import random  # Add this missing import
from datetime import datetime
from collections import deque
from colorama import Fore, Back, Style, init
from ultralytics import YOLO
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import warnings
import mediapipe as mp  # Add MediaPipe import

# Suppress MediaPipe warnings about feedback tensors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress matplotlib warnings
warnings.filterwarnings("ignore")

# Initialize colorama for colored terminal output
init(autoreset=True)

# Initialize pygame
pygame.init()

# Display parameters
WIDTH, HEIGHT = 1280, 720
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BRIGHT_GREEN = (0, 255, 128)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 128, 255)
TRANSPARENT_BLACK = (0, 0, 0, 150)  # Semi-transparent black

# Font setup
font_small = pygame.font.SysFont('Arial', 18)
font_medium = pygame.font.SysFont('Arial', 24)
font_large = pygame.font.SysFont('Arial', 32)

# HUD Parameters (initial values)
class HUDData:
    def __init__(self):
        self.altitude = 5000  # feet
        self.airspeed = 320   # knots
        self.heading = 45     # degrees
        self.pitch = 0        # degrees
        self.roll = 0         # degrees
        self.g_force = 1.0    # G
        self.aoa = 2.5        # Angle of attack (degrees)
        self.mach = 0.75      # Mach number
        self.fuel = 70        # Percentage
        self.weapon_status = "READY"
        self.target_distance = 12.5  # km
        self.target_locked = False
        self.waypoint_bearing = 85   # degrees
        self.waypoint_distance = 45  # km
        self.time = datetime.now()
        
        # Hand detection data
        self.hand_detected = False
        self.hand_position = (0, 0)
        self.hand_gesture = "None"
        self.num_hands = 0
        self.hand_confidence = 0.0
        self.hand_box = (0, 0, 0, 0)  # x1, y1, x2, y2
        self.hand_tracking_id = None
        self.dual_control = False
        self.prev_pitch = 0  # For G-force calculation
        
        # Flight dynamics - history for analytics
        self.altitude_history = deque(maxlen=100)
        self.airspeed_history = deque(maxlen=100)
        self.pitch_history = deque(maxlen=100)
        self.roll_history = deque(maxlen=100)
        
        # Event tracking
        self.last_event = None
        self.event_time = 0
        self.event_history = deque(maxlen=10)
        self.missile_warning = False
        self.terrain_warning = False
        self.system_faults = []

        # MediaPipe tracking data
        self.body_landmarks = None
        self.hand_landmarks = []
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.body_detected = False
        self.hand_gestures = {
            "left": "None",
            "right": "None"
        }

# Create semi-transparent surface
def create_transparent_surface(width, height, alpha=150):
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((0, 0, 0, alpha))
    return surface

# Hand Analyzer Class with GUI
class HandAnalyzer:
    def __init__(self, history_length=30):
        self.data_history = deque(maxlen=history_length)
        self.current_data = HUDData()
        self.prev_data = None
        self.history_length = history_length
        self.start_time = datetime.now()
        self.csv_filename = f"hand_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()
        
        # Initialize tkinter window
        self.root = tk.Tk()
        self.root.title("HUD Analytics Dashboard")
        self.root.geometry("1000x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup tabs
        self.tab_control = ttk.Notebook(self.root)
        
        # Hand detection tab
        self.tab_hands = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_hands, text="Hand Detection")
        
        # Flight data tab
        self.tab_flight = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_flight, text="Flight Parameters")
        
        # Status tab
        self.tab_status = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_status, text="System Status")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize plots
        self.setup_hand_detection_tab()
        self.setup_flight_data_tab()
        self.setup_status_tab()
        
        # Time series data for plots
        self.timestamps = deque(maxlen=100)
        self.hand_positions_x = deque(maxlen=100)
        self.hand_positions_y = deque(maxlen=100)
        self.hand_confidences = deque(maxlen=100)
        
    def init_csv(self):
        """Initialize CSV file with comprehensive headers"""
        try:
            # Create a directory for logs if it doesn't exist
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hud_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create CSV file path with timestamp and location
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_filename = os.path.join(log_dir, f"hud_data_{timestamp}.csv")
            
            # Extremely comprehensive headers for detailed data capture
            headers = [
                'timestamp', 
                # Hand detection data
                'hand_detected', 'num_hands', 'hand_position_x', 'hand_position_y',
                'hand_gesture', 'hand_confidence', 'hand_box_x1', 'hand_box_y1', 
                'hand_box_x2', 'hand_box_y2', 'dual_control', 'hand_tracking_id',
                'left_hand_gesture', 'right_hand_gesture',
                # Body tracking data
                'body_detected', 'head_position_x', 'head_position_y',
                # Flight parameters
                'altitude', 'airspeed', 'heading', 'pitch', 'roll', 'g_force', 
                'aoa', 'mach', 'fuel', 
                # Target information
                'target_locked', 'target_distance', 'weapon_status',
                # Navigation
                'waypoint_bearing', 'waypoint_distance',
                # Warnings and events
                'last_event', 'event_time', 'terrain_warning', 'missile_warning',
                # System data
                'detection_mode', 'fps',
                # Environment
                'session_runtime'
            ]
            
            with open(self.csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            
            print(f"\n[DATA] Enhanced CSV log created at: {os.path.abspath(self.csv_filename)}\n")
            
            # Initialize logging timer
            self.last_log_time = time.time()
            self.log_interval = 0.5  # Log every 0.5 seconds
            self.session_start_time = time.time()
            
            # Test write to ensure permissions are good
            self.log_test_record()
            
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            # Fall back to desktop or temp directory if home dir fails
            try:
                desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
                self.csv_filename = os.path.join(desktop_dir, f"hud_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                with open(self.csv_filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                print(f"Fallback CSV log created at: {self.csv_filename}")
            except:
                print("Could not create CSV file in any location. CSV logging disabled.")
                self.csv_filename = None

    def log_test_record(self):
        """Write a test record to verify CSV is working"""
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'TEST', 'TEST', 0, 0, 'TEST', 0.0, 0, 0, 0, 0, 0, 0, 'False',
                    'TEST', 0, 0.0, 0.0, 'False', 'False', 'TEST'
                ])
                file.flush()
            print("Test record written successfully to CSV")
        except Exception as e:
            print(f"Error writing test record: {e}")

    def log_to_csv(self):
        """Log current data to CSV with aggressive flushing for maximum data capture"""
        if not hasattr(self, 'csv_filename') or not self.csv_filename:
            return
            
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                
                # Get data, using defensive programming with defaults
                x, y = getattr(self.current_data, 'hand_position', (0, 0))
                
                # Get detection mode from global if available
                detection_mode = "Unknown"
                if hasattr(self.current_data, 'detection_mode'):
                    detection_mode = "MediaPipe" if self.current_data.detection_mode == 0 else "YOLO"
                
                # Comprehensive data row with all available metrics
                row = [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    getattr(self.current_data, 'hand_detected', False),
                    getattr(self.current_data, 'num_hands', 0),
                    x, y,
                    getattr(self.current_data, 'hand_gesture', "None"),
                    getattr(self.current_data, 'hand_confidence', 0.0),
                    getattr(self.current_data, 'altitude', 0),
                    getattr(self.current_data, 'airspeed', 0),
                    getattr(self.current_data, 'heading', 0),
                    getattr(self.current_data, 'pitch', 0),
                    getattr(self.current_data, 'roll', 0),
                    getattr(self.current_data, 'g_force', 1.0),
                    getattr(self.current_data, 'target_locked', False),
                    getattr(self.current_data, 'weapon_status', "UNKNOWN"),
                    getattr(self.current_data, 'fuel', 0),
                    getattr(self.current_data, 'mach', 0.0),
                    getattr(self.current_data, 'aoa', 0.0),
                    getattr(self.current_data, 'terrain_warning', False),
                    getattr(self.current_data, 'missile_warning', False),
                    detection_mode
                ]
                
                writer.writerow(row)
                
                # Force immediate flush to disk to prevent data loss
                file.flush()
                os.fsync(file.fileno())
                
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            # Try to recreate the file if it's been deleted or corrupted
            if not os.path.exists(self.csv_filename):
                print("CSV file missing, attempting to recreate...")
                self.init_csv()
    
    def flush_csv(self):
        """Force flush CSV file to disk"""
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                file.flush()
                os.fsync(file.fileno())
        except Exception as e:
            print(f"Error flushing CSV: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        print("Analytics window closed, data saved to CSV")
        # Don't actually close the window, as it would terminate the thread
        pass
    
    def setup_hand_detection_tab(self):
        """Setup the hand detection tab with plots"""
        # Hand position plot
        self.fig_hand_pos = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_hand_pos = self.fig_hand_pos.add_subplot(111)
        self.ax_hand_pos.set_title('Hand Position')
        self.ax_hand_pos.set_xlabel('X Position')
        self.ax_hand_pos.set_ylabel('Y Position')
        self.ax_hand_pos.set_xlim(0, WIDTH)
        self.ax_hand_pos.set_ylim(0, HEIGHT)
        self.ax_hand_pos.invert_yaxis()  # Invert Y axis to match screen coordinates
        self.scatter_hand_pos = self.ax_hand_pos.scatter([], [], c='red', alpha=0.5)
        
        self.hand_pos_canvas = FigureCanvasTkAgg(self.fig_hand_pos, self.tab_hands)
        self.hand_pos_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Hand detection frame
        self.hand_info_frame = ttk.LabelFrame(self.tab_hands, text="Hand Detection Info")
        self.hand_info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        self.hand_status_label = ttk.Label(self.hand_info_frame, text="Status: No hands detected", font=('Arial', 12))
        self.hand_status_label.pack(pady=5)
        
        self.hand_pos_label = ttk.Label(self.hand_info_frame, text="Position: (0, 0)", font=('Arial', 12))
        self.hand_pos_label.pack(pady=5)
        
        self.hand_gesture_label = ttk.Label(self.hand_info_frame, text="Gesture: None", font=('Arial', 12))
        self.hand_gesture_label.pack(pady=5)
        
        self.hand_conf_label = ttk.Label(self.hand_info_frame, text="Confidence: 0.0", font=('Arial', 12))
        self.hand_conf_label.pack(pady=5)
        
        # Confidence over time plot
        self.fig_conf = plt.Figure(figsize=(5, 2), dpi=100)
        self.ax_conf = self.fig_conf.add_subplot(111)
        self.ax_conf.set_title('Detection Confidence')
        self.ax_conf.set_xlabel('Time')
        self.ax_conf.set_ylabel('Confidence')
        self.ax_conf.set_ylim(0, 1)
        self.line_conf, = self.ax_conf.plot([], [], 'r-')
        
        self.conf_canvas = FigureCanvasTkAgg(self.fig_conf, self.tab_hands)
        self.conf_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
    
    def setup_flight_data_tab(self):
        """Setup the flight parameters tab with plots"""
        # Create figure with subplots
        self.fig_flight = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Altitude plot
        self.ax_altitude = self.fig_flight.add_subplot(221)
        self.ax_altitude.set_title('Altitude (ft)')
        self.ax_altitude.set_xlabel('Time')
        self.ax_altitude.set_ylabel('Altitude (ft)')
        self.line_altitude, = self.ax_altitude.plot([], [], 'g-')
        
        # Airspeed plot
        self.ax_airspeed = self.fig_flight.add_subplot(222)
        self.ax_airspeed.set_title('Airspeed (knots)')
        self.ax_airspeed.set_xlabel('Time')
        self.ax_airspeed.set_ylabel('Airspeed (knots)')
        self.line_airspeed, = self.ax_airspeed.plot([], [], 'b-')
        
        # Pitch plot
        self.ax_pitch = self.fig_flight.add_subplot(223)
        self.ax_pitch.set_title('Pitch (degrees)')
        self.ax_pitch.set_xlabel('Time')
        self.ax_pitch.set_ylabel('Pitch (degrees)')
        self.ax_pitch.set_ylim(-30, 30)
        self.line_pitch, = self.ax_pitch.plot([], [], 'r-')
        
        # Roll plot
        self.ax_roll = self.fig_flight.add_subplot(224)
        self.ax_roll.set_title('Roll (degrees)')
        self.ax_roll.set_xlabel('Time')
        self.ax_roll.set_ylabel('Roll (degrees)')
        self.ax_roll.set_ylim(-45, 45)
        self.line_roll, = self.ax_roll.plot([], [], 'y-')
        
        # Adjust layout
        self.fig_flight.tight_layout()
        
        # Add to tab
        self.flight_canvas = FigureCanvasTkAgg(self.fig_flight, self.tab_flight)
        self.flight_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    def setup_status_tab(self):
        """Setup the system status tab with event log"""
        # Create frame for status information
        self.status_frame = ttk.Frame(self.tab_status)
        self.status_frame.pack(fill=tk.BOTH, expand=1, padx=20, pady=20)
        
        # Time elapsed
        self.time_elapsed_label = ttk.Label(self.status_frame, text="Time Elapsed: 0s", font=('Arial', 14, 'bold'))
        self.time_elapsed_label.pack(pady=10)
        
        # Fuel gauge - using progressbar
        ttk.Label(self.status_frame, text="Fuel Level:", font=('Arial', 12)).pack(pady=5)
        self.fuel_gauge = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.fuel_gauge.pack(pady=10)
        
        # System status
        self.system_status_frame = ttk.LabelFrame(self.status_frame, text="System Status")
        self.system_status_frame.pack(fill=tk.X, pady=10)
        
        self.weapon_status_label = ttk.Label(self.system_status_frame, text="Weapon Status: READY", font=('Arial', 12))
        self.weapon_status_label.pack(pady=5)
        
        self.g_force_label = ttk.Label(self.system_status_frame, text="G-Force: 1.0G", font=('Arial', 12))
        self.g_force_label.pack(pady=5)
        
        self.aoa_label = ttk.Label(self.system_status_frame, text="Angle of Attack: 2.5°", font=('Arial', 12))
        self.aoa_label.pack(pady=5)
        
        self.mach_label = ttk.Label(self.system_status_frame, text="Mach: 0.75", font=('Arial', 12))
        self.mach_label.pack(pady=5)
        
        # Navigation info
        self.nav_frame = ttk.LabelFrame(self.status_frame, text="Navigation")
        self.nav_frame.pack(fill=tk.X, pady=10)
        
        self.heading_label = ttk.Label(self.nav_frame, text="Heading: 45°", font=('Arial', 12))
        self.heading_label.pack(pady=5)
        
        self.waypoint_label = ttk.Label(self.nav_frame, text="Waypoint: 45.0km / 085°", font=('Arial', 12))
        self.waypoint_label.pack(pady=5)
        
        self.target_label = ttk.Label(self.nav_frame, text="Target: Not Locked", font=('Arial', 12))
        self.target_label.pack(pady=5)
        
        # Event log
        self.event_frame = ttk.LabelFrame(self.status_frame, text="Event Log")
        self.event_frame.pack(fill=tk.X, pady=10)
        
        self.event_list = tk.Listbox(self.event_frame, height=6, font=('Courier', 10))
        self.event_list.pack(fill=tk.X, pady=5)
        
        # Warning indicators
        self.warning_frame = ttk.LabelFrame(self.status_frame, text="Warnings")
        self.warning_frame.pack(fill=tk.X, pady=10)
        
        self.missile_warning_label = ttk.Label(self.warning_frame, text="MISSILE WARNING: NONE", 
                                              font=('Arial', 12), foreground='gray')
        self.missile_warning_label.pack(pady=5)
        
        self.terrain_warning_label = ttk.Label(self.warning_frame, text="TERRAIN WARNING: NONE", 
                                              font=('Arial', 12), foreground='gray')
        self.terrain_warning_label.pack(pady=5)
        
    def update(self, hud_data):
        """Update with new HUD data"""
        # Store detection mode in the data object for CSV logging
        if not hasattr(hud_data, 'detection_mode'):
            hud_data.detection_mode = 0  # Default to MediaPipe
            
        # Track events for event log
        if hasattr(hud_data, 'last_event') and hud_data.last_event:
            if not self.prev_data or hud_data.last_event != self.prev_data.last_event:
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.event_list.insert(0, f"{timestamp} - {hud_data.last_event}")
                # Keep only last 20 events
                if self.event_list.size() > 20:
                    self.event_list.delete(20)
        
        # Update warning indicators
        if hasattr(hud_data, 'missile_warning') and hud_data.missile_warning:
            self.missile_warning_label.config(text="MISSILE WARNING: ACTIVE", foreground='red')
        else:
            self.missile_warning_label.config(text="MISSILE WARNING: NONE", foreground='gray')
            
        if hasattr(hud_data, 'terrain_warning') and hud_data.terrain_warning:
            self.terrain_warning_label.config(text="TERRAIN WARNING: PULL UP", foreground='red')
        else:
            self.terrain_warning_label.config(text="TERRAIN WARNING: NONE", foreground='gray')
        
        # Rest of the update method...
        self.prev_data = self.current_data
        self.current_data = hud_data
        self.data_history.append(hud_data)
        
        # Log data to CSV on EVERY update for maximum data capture
        self.log_to_csv()
        
        # No conditional flushing, always flush
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        # Still print status updates periodically
        if self.update_count % 100 == 0:
            print(f"Logged {self.update_count} records to CSV")
        
        # Update time series data
        self.timestamps.append(datetime.now())
        x, y = hud_data.hand_position
        self.hand_positions_x.append(x)
        self.hand_positions_y.append(y)
        self.hand_confidences.append(hud_data.hand_confidence if hud_data.hand_detected else 0)
        
        try:
            # Update hand position plot
            self.scatter_hand_pos.set_offsets(np.c_[self.hand_positions_x, self.hand_positions_y])
            
            # Update confidence plot
            times = range(len(self.hand_confidences))
            self.line_conf.set_data(times, self.hand_confidences)
            if times:
                self.ax_conf.set_xlim(0, max(times))
                self.ax_conf.figure.canvas.draw()
            
            # Update flight data plots
            times = range(len(hud_data.altitude_history))
            if times:
                self.line_altitude.set_data(times, list(hud_data.altitude_history))
                self.ax_altitude.set_xlim(0, max(times))
                self.ax_altitude.set_ylim(
                    min(hud_data.altitude_history) - 100, 
                    max(hud_data.altitude_history) + 100
                )
                
                self.line_airspeed.set_data(times, list(hud_data.airspeed_history))
                self.ax_airspeed.set_xlim(0, max(times))
                self.ax_airspeed.set_ylim(
                    min(hud_data.airspeed_history) - 10, 
                    max(hud_data.airspeed_history) + 10
                )
                
                self.line_pitch.set_data(times, list(hud_data.pitch_history))
                self.ax_pitch.set_xlim(0, max(times))
                
                self.line_roll.set_data(times, list(hud_data.roll_history))
                self.ax_roll.set_xlim(0, max(times))
                
                self.fig_flight.canvas.draw()
            
            # Update hand detection info
            if hud_data.hand_detected:
                self.hand_status_label.config(text=f"Status: Detected ({hud_data.num_hands} hands)")
                self.hand_pos_label.config(text=f"Position: ({x}, {y})")
                self.hand_gesture_label.config(text=f"Gesture: {hud_data.hand_gesture}")
                self.hand_conf_label.config(text=f"Confidence: {hud_data.hand_confidence:.2f}")
            else:
                self.hand_status_label.config(text="Status: No hands detected")
            
            # Update status tab
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.time_elapsed_label.config(text=f"Time Elapsed: {elapsed:.1f}s")
            
            self.fuel_gauge["value"] = hud_data.fuel
            self.weapon_status_label.config(text=f"Weapon Status: {hud_data.weapon_status}")
            self.g_force_label.config(text=f"G-Force: {hud_data.g_force:.1f}G")
            self.aoa_label.config(text=f"Angle of Attack: {hud_data.aoa:.1f}°")
            self.mach_label.config(text=f"Mach: {hud_data.mach:.2f}")
            
            self.heading_label.config(text=f"Heading: {int(hud_data.heading):03d}°")
            self.waypoint_label.config(text=f"Waypoint: {hud_data.waypoint_distance:.1f}km / {int(hud_data.waypoint_bearing):03d}°")
            
            if hud_data.target_locked:
                self.target_label.config(text=f"Target: LOCKED - {hud_data.target_distance:.1f}km")
            else:
                self.target_label.config(text="Target: Not Locked")
                
            # Update canvases
            self.hand_pos_canvas.draw()
            self.conf_canvas.draw()
            
            # Process tkinter events
            self.root.update()
            
        except Exception as e:
            print(f"Error updating analytics window: {e}")
    
    def run(self):
        """Run the tkinter main loop"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error in tkinter mainloop: {e}")

# HUD drawing functions with enhanced visuals
def draw_airspeed_indicator(screen, airspeed):
    # Create semi-transparent background
    airspeed_bg = create_transparent_surface(120, 320)
    
    # Draw border with gradient effect
    pygame.draw.rect(airspeed_bg, GREEN, (0, 0, 120, 320), 2)
    pygame.draw.line(airspeed_bg, BRIGHT_GREEN, (2, 2), (118, 2), 2)
    
    # Draw main indicator
    text = font_large.render(f"{int(airspeed)}", True, BRIGHT_GREEN)
    airspeed_bg.blit(text, (60 - text.get_width()//2, 160 - text.get_height()//2))
    
    text = font_small.render("KIAS", True, GREEN)
    airspeed_bg.blit(text, (60 - text.get_width()//2, 190))
    
    # Speed ticks with improved visibility
    for i in range(-5, 6):
        if i == 0:
            continue
        tick_speed = int(airspeed) + i * 20
        y_pos = 160 - i * 25
        
        # Draw tick mark
        pygame.draw.line(airspeed_bg, GREEN, (10, y_pos), (30, y_pos), 2)
        
        # Draw speed value
        text = font_small.render(f"{tick_speed}", True, GREEN)
        airspeed_bg.blit(text, (40, y_pos-10))
    
    # Blit the entire airspeed indicator to screen
    screen.blit(airspeed_bg, (40, HEIGHT//2-160))

def draw_altitude_indicator(screen, altitude):
    # Create semi-transparent background
    altitude_bg = create_transparent_surface(120, 320)
    
    # Draw border with gradient effect
    pygame.draw.rect(altitude_bg, GREEN, (0, 0, 120, 320), 2)
    pygame.draw.line(altitude_bg, BRIGHT_GREEN, (2, 2), (118, 2), 2)
    
    # Draw main indicator
    text = font_large.render(f"{int(altitude)}", True, BRIGHT_GREEN)
    altitude_bg.blit(text, (60 - text.get_width()//2, 160 - text.get_height()//2))
    
    text = font_small.render("ALT", True, GREEN)
    altitude_bg.blit(text, (60 - text.get_width()//2, 190))
    
    # Altitude ticks with improved visibility
    for i in range(-5, 6):
        if i == 0:
            continue
        tick_alt = int(altitude) + i * 500
        y_pos = 160 - i * 25
        
        # Draw tick mark
        pygame.draw.line(altitude_bg, GREEN, (90, y_pos), (110, y_pos), 2)
        
        # Draw altitude value
        text = font_small.render(f"{tick_alt}", True, GREEN)
        altitude_bg.blit(text, (60 - text.get_width(), y_pos-10))
    
    # Blit the entire altitude indicator to screen
    screen.blit(altitude_bg, (WIDTH-160, HEIGHT//2-160))

def draw_heading_indicator(screen, heading):
    center_x = WIDTH // 2
    
    # Create semi-transparent background
    heading_bg = create_transparent_surface(350, 100)
    
    # Draw border
    pygame.draw.rect(heading_bg, GREEN, (0, 0, 350, 50), 2)
    pygame.draw.line(heading_bg, GREEN, (175, 0), (175, 20), 2)
    
    # Draw heading value with enhanced visibility
    text = font_medium.render(f"{int(heading):03d}°", True, BRIGHT_GREEN)
    heading_bg.blit(text, (175 - text.get_width()//2, 5))
    
    # Draw heading ticks with improved clarity
    for i in range(-4, 5):
        tick_heading = (heading + i * 10) % 360
        x_pos = 175 + i * 30
        
        if 0 <= x_pos <= 350:
            # Draw tick
            pygame.draw.line(heading_bg, GREEN, (x_pos, 40), (x_pos, 50), 2)
            
            # Draw heading value at tick
            text = font_small.render(f"{int(tick_heading):03d}", True, GREEN)
            heading_bg.blit(text, (x_pos - text.get_width()//2, 50))
    
    # Blit heading indicator to screen
    screen.blit(heading_bg, (center_x-175, 30))

def draw_horizon(screen, pitch, roll):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    horizon_width = 700
    
    # Calculate horizon line based on pitch and roll
    roll_rad = math.radians(roll)
    pitch_offset = pitch * 5  # 5 pixels per degree of pitch
    
    # Calculate sky and ground colors with alpha
    sky_surface = create_transparent_surface(WIDTH, HEIGHT, alpha=50)
    sky_surface.fill((0, 50, 100, 50))  # Transparent blue for sky
    
    ground_surface = create_transparent_surface(WIDTH, HEIGHT, alpha=50)
    ground_surface.fill((50, 30, 0, 50))  # Transparent brown for ground
    
    # Calculate horizon endpoints
    x1 = center_x - (horizon_width/2) * math.cos(roll_rad)
    y1 = center_y + (horizon_width/2) * math.sin(roll_rad) - pitch_offset
    x2 = center_x + (horizon_width/2) * math.cos(roll_rad)
    y2 = center_y - (horizon_width/2) * math.sin(roll_rad) - pitch_offset
    
    # Draw artificial horizon
    pygame.draw.line(screen, BRIGHT_GREEN, (int(x1), int(y1)), (int(x2), int(y2)), 3)
    
    # Draw roll indicator at the top
    roll_ind_radius = 50
    roll_ind_x = center_x
    roll_ind_y = center_y - 200
    
    # Draw roll circle
    pygame.draw.circle(screen, GREEN, (roll_ind_x, roll_ind_y), roll_ind_radius, 1)
    
    # Draw roll indicator needle
    roll_needle_x = roll_ind_x + roll_ind_radius * math.sin(roll_rad)
    roll_needle_y = roll_ind_y - roll_ind_radius * math.cos(roll_rad)
    pygame.draw.line(screen, BRIGHT_GREEN, (roll_ind_x, roll_ind_y), 
                    (int(roll_needle_x), int(roll_needle_y)), 2)
    
    # Draw roll indicator markers
    for angle in range(0, 361, 30):
        angle_rad = math.radians(angle)
        marker_x = roll_ind_x + roll_ind_radius * math.sin(angle_rad)
        marker_y = roll_ind_y - roll_ind_radius * math.cos(angle_rad)
        
        # Draw longer line for 0, 90, 180, 270 degrees
        if angle % 90 == 0:
            pygame.draw.line(screen, GREEN, 
                            (int(roll_ind_x + (roll_ind_radius-10) * math.sin(angle_rad)),
                             int(roll_ind_y - (roll_ind_radius-10) * math.cos(angle_rad))),
                            (int(marker_x), int(marker_y)), 2)
        else:
            pygame.draw.line(screen, GREEN, 
                            (int(roll_ind_x + (roll_ind_radius-5) * math.sin(angle_rad)),
                             int(roll_ind_y - (roll_ind_radius-5) * math.cos(angle_rad))),
                            (int(marker_x), int(marker_y)), 1)
    
    # Draw pitch ladder with improved visibility
    for p in range(-20, 21, 5):
        if p == 0:
            continue
        
        ladder_width = 120 if p % 10 == 0 else 60
        pitch_pixel = p * 5
        
        x1 = center_x - (ladder_width/2) * math.cos(roll_rad)
        y1 = center_y + (ladder_width/2) * math.sin(roll_rad) - pitch_offset - pitch_pixel
        x2 = center_x + (ladder_width/2) * math.cos(roll_rad)
        y2 = center_y - (ladder_width/2) * math.sin(roll_rad) - pitch_offset - pitch_pixel
        
        if -200 < y1 < HEIGHT+200 and -200 < y2 < HEIGHT+200:
            color = BRIGHT_GREEN if p % 10 == 0 else GREEN
            pygame.draw.line(screen, color, (int(x1), int(y1)), (int(x2), int(y2)), 2)
            
            # Add a pitch value
            text = font_small.render(f"{abs(p)}", True, color)
            text_x = int(x2) + 15
            text_y = int(y2) - 10
            screen.blit(text, (text_x, text_y))

def draw_flight_path_vector(screen, pitch, roll):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    
    # Draw flight path vector (FPV) symbol with enhanced design
    pygame.draw.circle(screen, BRIGHT_GREEN, (center_x, center_y), 12, 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (center_x-20, center_y), (center_x-10, center_y), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (center_x+10, center_y), (center_x+20, center_y), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (center_x, center_y-20), (center_x, center_y-10), 2)
    
    # Add small diagonal lines to represent velocity vector wings
    pygame.draw.line(screen, BRIGHT_GREEN, (center_x-15, center_y-5), (center_x-8, center_y), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (center_x+15, center_y-5), (center_x+8, center_y), 2)

def draw_status_indicators(screen, data):
    # G-force
    text = font_medium.render(f"G: {data.g_force:.1f}", True, GREEN)
    screen.blit(text, (50, HEIGHT-80))
    
    # AOA
    text = font_medium.render(f"AoA: {data.aoa:.1f}°", True, GREEN)
    screen.blit(text, (50, HEIGHT-50))
    
    # Mach number
    text = font_medium.render(f"MACH: {data.mach:.2f}", True, GREEN)
    screen.blit(text, (50, HEIGHT-110))
    
    # Fuel
    text = font_medium.render(f"FUEL: {data.fuel}%", True, GREEN if data.fuel > 20 else RED)
    screen.blit(text, (WIDTH-200, HEIGHT-50))
    
    # Weapon status
    text = font_medium.render(f"WPN: {data.weapon_status}", True, GREEN)
    screen.blit(text, (WIDTH-200, HEIGHT-80))
    
    # Time
    time_str = data.time.strftime("%H:%M:%S")
    text = font_medium.render(time_str, True, GREEN)
    screen.blit(text, (WIDTH//2-50, HEIGHT-50))

def draw_target_info(screen, data):
    if data.target_locked:
        # Draw target diamond
        pygame.draw.polygon(screen, RED, [
            (WIDTH//2+100, HEIGHT//2-100),
            (WIDTH//2+120, HEIGHT//2-80),
            (WIDTH//2+100, HEIGHT//2-60),
            (WIDTH//2+80, HEIGHT//2-80),
        ], 2)
        
        # Target distance
        text = font_medium.render(f"TGT: {data.target_distance:.1f} KM", True, RED)
        screen.blit(text, (WIDTH//2+130, HEIGHT//2-90))
    
    # Waypoint
    text = font_medium.render(f"WPT: {data.waypoint_distance:.1f}KM / {int(data.waypoint_bearing):03d}°", True, GREEN)
    screen.blit(text, (WIDTH//2+50, 100))

def draw_hand_detection_info(screen, data):
    """Draw hand detection information on HUD"""
    if data.hand_detected:
        # Draw hand position indicator with crosshair
        x, y = data.hand_position
        
        # Draw animated targeting circle
        time_factor = time.time() % 1.0  # 0.0 to 1.0 oscillation
        radius = 20 + int(5 * math.sin(time_factor * 6.28))
        
        # Draw concentric circles
        pygame.draw.circle(screen, RED, (x, y), radius, 2)
        pygame.draw.circle(screen, RED, (x, y), radius//2, 1)
        
        # Draw crosshair
        pygame.draw.line(screen, RED, (x-radius, y), (x+radius, y), 2)
        pygame.draw.line(screen, RED, (x, y-radius), (x, y+radius), 2)
        
        # Display hand info in a small transparent panel
        info_bg = create_transparent_surface(150, 70)
        pygame.draw.rect(info_bg, RED, (0, 0, 150, 70), 1)
        
        # Hand gesture info
        text = font_medium.render(f"{data.hand_gesture}", True, RED)
        info_bg.blit(text, (10, 10))
        
        # Confidence info
        text = font_small.render(f"CONF: {data.hand_confidence:.2f}", True, RED)
        info_bg.blit(text, (10, 40))
        
        # Position the info panel to avoid screen edges
        info_x = min(max(x + 30, 20), WIDTH - 170)
        info_y = min(max(y - 30, 20), HEIGHT - 90)
        screen.blit(info_bg, (info_x, info_y))

def update_hud_values(data):
    """Update HUD values with random changes"""
    data.altitude += np.random.uniform(-20, 20)
    data.airspeed += np.random.uniform(-2, 2)
    data.heading = (data.heading + np.random.uniform(-0.5, 0.5)) % 360
    data.pitch = max(-30, min(30, data.pitch + np.random.uniform(-0.3, 0.3)))
    data.roll = max(-45, min(45, data.roll + np.random.uniform(-0.5, 0.5)))
    data.g_force = max(0.1, min(9.0, data.g_force + np.random.uniform(-0.05, 0.05)))
    data.aoa = max(-10, min(20, data.aoa + np.random.uniform(-0.1, 0.1)))
    data.mach = max(0.5, min(1.2, data.mach + np.random.uniform(-0.01, 0.01)))
    data.fuel = max(0, min(100, data.fuel - np.random.uniform(0.01, 0.05)))
    
    if np.random.random() < 0.01:
        data.target_locked = not data.target_locked
    
    if data.target_locked:
        data.target_distance = max(0.5, min(25, data.target_distance + np.random.uniform(-0.2, 0.2)))
    
    data.waypoint_distance = max(0, data.waypoint_distance - np.random.uniform(0.05, 0.1))
    data.waypoint_bearing = (data.waypoint_bearing + np.random.uniform(-0.2, 0.2)) % 360
    data.time = datetime.now()
    
    return data

def analyze_hand_movement(data):
    """Analyze hand movement and update HUD parameters based on hand tilt"""
    if data.hand_detected:
        x, y = data.hand_position
        x1, y1, x2, y2 = data.hand_box
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        
        # Calculate hand tilt
        hand_width = x2 - x1
        hand_height = y2 - y1
        
        # Calculate hand orientation for roll
        # If hand is tilted, use that to control roll
        hand_angle = 0
        if hand_height != 0:
            hand_angle = math.degrees(math.atan2(hand_width, hand_height))
            # Map hand angle to roll (-45 to 45 degrees)
            data.roll = max(-45, min(45, hand_angle * 2))
        
        # Use vertical position for pitch control
        # Higher hand = nose up, lower hand = nose down
        pitch_factor = (center_y - y) / (HEIGHT / 3)  # -1 to 1 range
        data.pitch = max(-30, min(30, pitch_factor * 20))
        
        # Gesture recognition for special actions
        if data.hand_gesture == "Open Palm":
            # Increase speed with open palm
            data.airspeed += 0.8
            data.mach += 0.001
            
        elif data.hand_gesture == "Closed":
            # Decrease speed with closed fist
            data.airspeed -= 0.5
            data.mach -= 0.0005
            
        elif data.hand_gesture == "Pointing":
            # Target lock with pointing gesture
            if not data.target_locked and np.random.random() < 0.1:
                data.target_locked = True
                data.target_distance = np.random.uniform(5, 15)
                data.weapon_status = "LOCK"
        
        # Update G-force based on pitch change rate
        if hasattr(data, 'prev_pitch'):
            pitch_change = abs(data.pitch - data.prev_pitch)
            data.g_force = min(9.0, max(0.1, 1.0 + pitch_change / 2))
        data.prev_pitch = data.pitch
        
        # Random events based on hand position
        if np.random.random() < 0.005:  # 0.5% chance per frame
            # Random system events
            events = [
                "RADAR CONTACT",
                "MISSILE ALERT",
                "TERRAIN WARNING",
                "FUEL LOW",
                "SYSTEM FAULT",
                "TARGET ACQUIRED"
            ]
            data.last_event = random.choice(events)
            data.event_time = time.time()
            
            # Handle specific events
            if data.last_event == "MISSILE ALERT":
                data.missile_warning = True
            elif data.last_event == "TARGET ACQUIRED":
                data.target_locked = True
                data.target_distance = np.random.uniform(5, 15)
    else:
        # Gradual return to level flight when no hands detected
        data.roll *= 0.95
        data.pitch *= 0.95
        
    return data

def detect_hands(frame, model):
    """Detect hands in frame using YOLO model with improved orientation analysis"""
    results = model(frame, classes=[0])  # Focus on people class (0) which includes hands
    
    # Create a new HUD data object
    data = HUDData()
    
    # Update basic HUD values and store in history
    data = update_hud_values(data)
    data.altitude_history.append(data.altitude)
    data.airspeed_history.append(data.airspeed)
    data.pitch_history.append(data.pitch)
    data.roll_history.append(data.roll)
    
    # Process YOLO results
    if results and len(results) > 0:
        result = results[0]  # Get first result
        boxes = result.boxes
        
        if len(boxes) > 0:
            data.hand_detected = True
            data.num_hands = len(boxes)
            
            # Get the first hand detection (highest confidence)
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Calculate center point of the hand
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            data.hand_position = (center_x, center_y)
            data.hand_confidence = confidence
            data.hand_box = (x1, y1, x2, y2)
            
            # Enhanced hand gesture recognition
            hand_width = x2 - x1
            hand_height = y2 - y1
            aspect_ratio = hand_width / hand_height if hand_height > 0 else 1
            area = hand_width * hand_height
            
            if aspect_ratio > 1.5:
                data.hand_gesture = "Open Palm"
            elif aspect_ratio < 0.6:
                data.hand_gesture = "Pointing"
            else:
                if area < 5000:  # Small area likely means closed fist
                    data.hand_gesture = "Closed"
                else:
                    data.hand_gesture = "Unknown"
            
            # Draw enhanced visualization on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add orientation line to visualize hand tilt
            cv2.line(frame, 
                    (center_x, center_y), 
                    (center_x + int(hand_width/2), center_y + int(hand_height/2)),
                    (255, 0, 255), 2)
            
            # Add text with gesture and status
            gesture_text = f"{data.hand_gesture} ({confidence:.2f})"
            cv2.putText(frame, gesture_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If we have multiple hands, process them for additional control
            if len(boxes) > 1:
                data.dual_control = True
                # Process second hand for additional controls
                box2 = boxes[1]
                x1_2, y1_2, x2_2, y2_2 = map(int, box2.xyxy[0])
                center_x_2 = (x1_2 + x2_2) // 2
                center_y_2 = (y1_2 + y2_2) // 2
                
                # Distance between hands can control throttle
                hand_distance = math.sqrt((center_x_2 - center_x)**2 + (center_y_2 - center_y)**2)
                # Map distance to airspeed boost
                if hand_distance > 200:
                    data.airspeed += 1.0  # Boost speed more with wide hand separation
                
                cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), (0, 255, 255), 2)
                cv2.line(frame, (center_x, center_y), (center_x_2, center_y_2), (255, 0, 0), 2)
            
    # Analyze hand movement effects on flight parameters
    data = analyze_hand_movement(data)
    
    return frame, data

def draw_enhanced_hud(screen, frame_surface, data):
    """Draw all HUD elements with enhanced visuals and events"""
    # First blit the camera frame
    screen.blit(frame_surface, (0, 0))
    
    # Draw main HUD elements
    draw_airspeed_indicator(screen, data.airspeed)
    draw_altitude_indicator(screen, data.altitude)
    draw_heading_indicator(screen, data.heading)
    draw_horizon(screen, data.pitch, data.roll)
    draw_flight_path_vector(screen, data.pitch, data.roll)
    draw_status_indicators(screen, data)
    draw_target_info(screen, data)
    
    # Draw tracking visualizations if enabled
    if data.hand_detected:
        draw_hand_detection_info(screen, data)
    
    if data.body_detected:
        draw_body_tracking_info(screen, data)
    
    # Draw event notifications
    if data.last_event and time.time() - data.event_time < 3:  # Show for 3 seconds
        # Create flashing effect
        if int(time.time() * 2) % 2 == 0:  # Flash at 2Hz
            event_bg = create_transparent_surface(400, 80, alpha=200)
            pygame.draw.rect(event_bg, RED if "ALERT" in data.last_event else YELLOW, (0, 0, 400, 80), 3)
            
            text = font_large.render(data.last_event, True, RED if "ALERT" in data.last_event else YELLOW)
            event_bg.blit(text, (200 - text.get_width()//2, 25))
            
            screen.blit(event_bg, (WIDTH//2 - 200, 150))
    
    # Draw missile warning
    if hasattr(data, 'missile_warning') and data.missile_warning:
        if int(time.time() * 4) % 2 == 0:  # Flash at 4Hz
            text = font_large.render("MISSILE WARNING", True, RED)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, 200))
    
    # Draw terrain warning if needed
    if hasattr(data, 'terrain_warning') and data.terrain_warning:
        if int(time.time() * 4) % 2 == 0:  # Flash at 4Hz
            text = font_large.render("PULL UP", True, RED)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, 250))
    
    # Draw HUD frame - adds a subtle frame around the entire display
    pygame.draw.rect(screen, GREEN, (0, 0, WIDTH, HEIGHT), 1)
    
    # Add small corner brackets for aesthetic
    bracket_size = 30
    # Top-left
    pygame.draw.line(screen, BRIGHT_GREEN, (0, 0), (bracket_size, 0), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (0, 0), (0, bracket_size), 2)
    # Top-right
    pygame.draw.line(screen, BRIGHT_GREEN, (WIDTH-bracket_size, 0), (WIDTH, 0), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (WIDTH, 0), (WIDTH, bracket_size), 2)
    # Bottom-left
    pygame.draw.line(screen, BRIGHT_GREEN, (0, HEIGHT), (bracket_size, HEIGHT), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (0, HEIGHT-bracket_size), (0, HEIGHT), 2)
    # Bottom-right
    pygame.draw.line(screen, BRIGHT_GREEN, (WIDTH-bracket_size, HEIGHT), (WIDTH, HEIGHT), 2)
    pygame.draw.line(screen, BRIGHT_GREEN, (WIDTH, HEIGHT-bracket_size), (WIDTH, HEIGHT), 2)
    
    # Draw detection mode indicator
    if data.hand_detected:
        text = font_medium.render("HAND CONTROL ACTIVE", True, BRIGHT_GREEN)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 100))
        
        if data.dual_control:
            text = font_small.render("DUAL HAND MODE", True, BRIGHT_GREEN)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 75))
            
    if data.body_detected:
        text = font_small.render("BODY TRACKING ACTIVE", True, BLUE)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 50))

def draw_body_tracking_info(screen, data):
    """Draw body tracking visualization"""
    if not data.body_detected or not data.body_landmarks:
        return
    
    # Create transparent overlay for body stats
    body_info = create_transparent_surface(200, 100)
    pygame.draw.rect(body_info, BLUE, (0, 0, 200, 100), 1)
    
    # Add body tracking status
    text = font_small.render("Pose Tracking", True, BLUE)
    body_info.blit(text, (10, 10))
    
    # Could add more body metrics here
    # For example, estimate if pilot is looking forward
    text = font_small.render("Pilot Status: Active", True, BLUE)
    body_info.blit(text, (10, 40))
    
    # Position the info panel
    screen.blit(body_info, (WIDTH - 220, 100))

def init_mediapipe():
    """Initialize MediaPipe pose and hands modules"""
    # Hands setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Pose setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    
    return hands, pose, mp_drawing, mp_hands, mp_pose

def detect_with_mediapipe(frame, hands_model, pose_model, mp_drawing, mp_hands, mp_pose):
    """Detect hands and body pose using MediaPipe"""
    # Create a new HUD data object
    data = HUDData()
    
    # Update basic HUD values and store in history
    data = update_hud_values(data)
    data.altitude_history.append(data.altitude)
    data.airspeed_history.append(data.airspeed)
    data.pitch_history.append(data.pitch)
    data.roll_history.append(data.roll)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Hands
    hands_results = hands_model.process(rgb_frame)
    
    # Process with MediaPipe Pose
    pose_results = pose_model.process(rgb_frame)
    
    # Store body landmarks if detected
    if pose_results.pose_landmarks:
        data.body_detected = True
        data.body_landmarks = pose_results.pose_landmarks
        
        # Draw body landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, 
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=1)
        )
    
    # Process hand landmarks if detected
    if hands_results.multi_hand_landmarks:
        data.hand_detected = True
        data.num_hands = len(hands_results.multi_hand_landmarks)
        data.hand_landmarks = hands_results.multi_hand_landmarks
        
        # Process each hand
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            # Get hand classification (left/right)
            handedness = hands_results.multi_handedness[idx].classification[0].label
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
            
            # Store landmarks by hand type
            if handedness == "Left":
                data.left_hand_landmarks = hand_landmarks
            else:
                data.right_hand_landmarks = hand_landmarks
            
            # Calculate hand center (using wrist as reference)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * frame.shape[1])
            y = int(wrist.y * frame.shape[0])
            
            # Set hand position to main hand (first detected)
            if idx == 0:
                data.hand_position = (x, y)
                
                # Calculate bounding box
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                x1 = int(min(x_coords) * frame.shape[1])
                y1 = int(min(y_coords) * frame.shape[0])
                x2 = int(max(x_coords) * frame.shape[1])
                y2 = int(max(y_coords) * frame.shape[0])
                data.hand_box = (x1, y1, x2, y2)
                
                # Set confidence (MediaPipe doesn't provide confidence values like YOLO)
                data.hand_confidence = 0.9  # Default high confidence for detected landmarks
                
            # Detect hand gesture based on finger positions
            gesture = detect_hand_gesture(hand_landmarks, mp_hands)
            
            if handedness == "Left":
                data.hand_gestures["left"] = gesture
            else:
                data.hand_gestures["right"] = gesture
            
            # Use the first detected hand's gesture as the main gesture
            if idx == 0:
                data.hand_gesture = gesture
            
            # Add text with gesture and handedness
            cv2.putText(frame, f"{handedness}: {gesture}", (x-10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # If no hands detected, make sure we indicate that
    else:
        data.hand_detected = False
        data.num_hands = 0
    
    # Analyze movement effects on flight parameters
    data = analyze_mediapipe_movement(data)
    
    return frame, data

def detect_hand_gesture(landmarks, mp_hands):
    """Detect hand gesture from landmarks"""
    # Extract key finger landmarks
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get MCP (knuckle) positions for reference
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if fingers are extended (above their MCP)
    thumb_extended = thumb_tip.x < wrist.x if thumb_tip.x < wrist.x else thumb_tip.x > wrist.x
    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_mcp.y
    
    # Define gestures based on finger positions
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing"
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Open Palm"
    elif not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Closed"
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Victory"
    elif index_extended and middle_extended and ring_extended and not pinky_extended:
        return "Three"
    elif index_extended and pinky_extended and not middle_extended and not ring_extended:
        return "Rock"
    else:
        return "Unknown"

def analyze_mediapipe_movement(data):
    """Analyze MediaPipe landmarks to control HUD parameters"""
    if data.hand_detected:
        x, y = data.hand_position
        x1, y1, x2, y2 = data.hand_box
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        
        # Calculate hand tilt if we have landmarks
        if data.hand_landmarks and len(data.hand_landmarks) > 0:
            # Use the first hand for primary control
            landmarks = data.hand_landmarks[0]
            
            # Calculate orientation using wrist and middle finger MCP
            wrist_idx = 0  # MediaPipe Hands wrist index
            middle_mcp_idx = 9  # MediaPipe Hands middle finger MCP index
            
            wrist = landmarks.landmark[wrist_idx]
            middle_mcp = landmarks.landmark[middle_mcp_idx]
            
            # Calculate angle for roll
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            hand_angle = math.degrees(math.atan2(dy, dx))
            
            # Map hand rotation to roll (-45 to 45 degrees)
            # Adjust the mapping as needed for intuitive control
            data.roll = max(-45, min(45, hand_angle))
            
            # Use vertical position for pitch control
            # Higher hand = nose up, lower hand = nose down
            pitch_factor = (center_y - y) / (HEIGHT / 3)  # -1 to 1 range
            data.pitch = max(-30, min(30, pitch_factor * 20))
        
        # Gesture-based controls
        main_gesture = data.hand_gesture
        
        if main_gesture == "Open Palm":
            # Increase speed with open palm
            data.airspeed += 0.8
            data.mach += 0.001
            
        elif main_gesture == "Closed":
            # Decrease speed with closed fist
            data.airspeed -= 0.5
            data.mach -= 0.0005
            
        elif main_gesture == "Pointing":
            # Target lock with pointing gesture
            if not data.target_locked and np.random.random() < 0.1:
                data.target_locked = True
                data.target_distance = np.random.uniform(5, 15)
                data.weapon_status = "LOCK"
                
        elif main_gesture == "Victory":
            # Toggle special mode or weapon status
            if np.random.random() < 0.05:  # Occasionally change weapon status
                data.weapon_status = random.choice(["ARMED", "READY", "STANDBY"])
        
        # Dual hand controls if both hands detected
        if data.num_hands > 1:
            data.dual_control = True
            # Two-handed gesture combinations could be added here
        
        # Update G-force based on pitch change rate
        if hasattr(data, 'prev_pitch'):
            pitch_change = abs(data.pitch - data.prev_pitch)
            data.g_force = min(9.0, max(0.1, 1.0 + pitch_change / 2))
        data.prev_pitch = data.pitch
        
        # Random events based on hand position
        if np.random.random() < 0.005:  # 0.5% chance per frame
            # Random system events
            events = [
                "RADAR CONTACT",
                "MISSILE ALERT",
                "TERRAIN WARNING",
                "FUEL LOW",
                "SYSTEM FAULT",
                "TARGET ACQUIRED"
            ]
            data.last_event = random.choice(events)
            data.event_time = time.time()
            
            # Handle specific events
            if data.last_event == "MISSILE ALERT":
                data.missile_warning = True
            elif data.last_event == "TARGET ACQUIRED":
                data.target_locked = True
                data.target_distance = np.random.uniform(5, 15)
            elif data.last_event == "TERRAIN WARNING":
                data.terrain_warning = True
                
    else:
        # Gradual return to level flight when no hands detected
        data.roll *= 0.95
        data.pitch *= 0.95
        
        # Clear warnings over time
        if hasattr(data, 'missile_warning') and data.missile_warning and np.random.random() < 0.01:
            data.missile_warning = False
        if hasattr(data, 'terrain_warning') and data.terrain_warning and np.random.random() < 0.01:
            data.terrain_warning = False
        
    return data

def main():
    """Main function to run the camera-based HUD with hand detection"""
    # Load YOLO model for hand detection as backup
    print("Loading YOLO model for hand detection...")
    yolo_model = YOLO("yolov8n.pt")  # Use yolov8 nano model
    
    # Initialize MediaPipe
    print("Initializing MediaPipe...")
    hands, pose, mp_drawing, mp_hands, mp_pose = init_mediapipe()
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use default camera (0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Create pygame display
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fighter Jet HUD with MediaPipe Tracking")
    
    # Initialize HUD data and analyzer
    hud_data = HUDData()
    # Use a list to allow sharing mutable data between threads
    hud_data_ref = [hud_data]
    analyzer = HandAnalyzer()
    
    # Start analyzer thread
    analyzer_thread = threading.Thread(target=run_analyzer, args=(analyzer, hud_data_ref), daemon=True)
    analyzer_thread.start()
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    prev_frame_time = 0
    new_frame_time = 0
    
    # Detection mode (0=MediaPipe, 1=YOLO)
    detection_mode = 0
    
    print("HUD system ready. Press ESC to exit, M to switch detection mode.")
    
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Add special key for toggling target lock manually
                elif event.key == pygame.K_t:
                    hud_data_ref[0].target_locked = not hud_data_ref[0].target_locked
                # Toggle between detection modes
                elif event.key == pygame.K_m:
                    detection_mode = 1 - detection_mode
                    mode_name = "YOLO" if detection_mode == 1 else "MediaPipe"
                    print(f"Switched to {mode_name} detection mode")
                # Add manual controls for testing
                elif event.key == pygame.K_UP:
                    hud_data_ref[0].pitch += 2
                elif event.key == pygame.K_DOWN:
                    hud_data_ref[0].pitch -= 2
                elif event.key == pygame.K_LEFT:
                    hud_data_ref[0].roll -= 5
                elif event.key == pygame.K_RIGHT:
                    hud_data_ref[0].roll += 5
        
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 60
        prev_frame_time = new_frame_time
        fps = int(fps)
        
        # Add FPS display
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (100, 255, 0), 2, cv2.LINE_AA)
        
        # Show active detection mode
        mode_name = "MediaPipe" if detection_mode == 0 else "YOLO"
        cv2.putText(frame, f"Mode: {mode_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (100, 255, 0), 2, cv2.LINE_AA)
        
        # Run hand detection based on selected mode
        if detection_mode == 0:
            # Use MediaPipe
            frame, hud_data = detect_with_mediapipe(frame, hands, pose, mp_drawing, mp_hands, mp_pose)
            hud_data.detection_mode = 0
        else:
            # Use YOLO
            frame, hud_data = detect_hands(frame, yolo_model)
            hud_data.detection_mode = 1
        
        # Update the shared data
        hud_data_ref[0] = hud_data
        
        # Convert frame to pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        
        # Draw the enhanced HUD overlay
        draw_enhanced_hud(screen, frame_surface, hud_data)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS
    
    # Clean up
    cap.release()
    hands.close()
    pose.close()
    pygame.quit()
    print("HUD system terminated.")

def run_analyzer(analyzer, hud_data_ref):
    """Run the analyzer in a separate thread"""
    try:
        while True:
            # Update the analyzer with the latest HUD data
            if hud_data_ref and len(hud_data_ref) > 0:
                analyzer.update(hud_data_ref[0])
            time.sleep(0.1)  # Update at 10Hz to avoid overloading
    except Exception as e:
        print(f"Error in analyzer thread: {e}")

if __name__ == "__main__":
    main()