import pygame
import cv2
import numpy as np
import math
import time
import os
import csv
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
        
        # Flight dynamics - history for analytics
        self.altitude_history = deque(maxlen=100)
        self.airspeed_history = deque(maxlen=100)
        self.pitch_history = deque(maxlen=100)
        self.roll_history = deque(maxlen=100)

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
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'hand_detected', 'num_hands', 'hand_position_x', 
            'hand_position_y', 'hand_gesture', 'hand_confidence', 'altitude', 
            'airspeed', 'heading', 'pitch', 'roll', 'g_force', 'target_locked'
        ]
        
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    def log_to_csv(self):
        """Log current data to CSV"""
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            x, y = self.current_data.hand_position
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                self.current_data.hand_detected,
                self.current_data.num_hands,
                x, y,
                self.current_data.hand_gesture,
                self.current_data.hand_confidence,
                self.current_data.altitude,
                self.current_data.airspeed,
                self.current_data.heading,
                self.current_data.pitch,
                self.current_data.roll,
                self.current_data.g_force,
                self.current_data.target_locked
            ])
            
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
        """Setup the system status tab"""
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
        
    def update(self, hud_data):
        """Update with new HUD data"""
        self.prev_data = self.current_data
        self.current_data = hud_data
        self.data_history.append(hud_data)
        
        # Log data to CSV
        self.log_to_csv()
        
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
    """Analyze hand movement and update HUD parameters based on gesture/position"""
    if data.hand_detected:
        x, y = data.hand_position
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        
        # If hand is on the left side of screen, adjust roll
        if x < center_x - 200:
            data.roll = max(-45, data.roll - 0.5)
        # If hand is on the right side of screen, adjust roll
        elif x > center_x + 200:
            data.roll = min(45, data.roll + 0.5)
        # Otherwise gradually return to level
        else:
            data.roll = data.roll * 0.95
        
        # If hand is above center, increase pitch
        if y < center_y - 150:
            data.pitch = min(30, data.pitch + 0.3)
        # If hand is below center, decrease pitch
        elif y > center_y + 150:
            data.pitch = max(-30, data.pitch - 0.3)
        # Otherwise gradually return to level
        else:
            data.pitch = data.pitch * 0.95
        
        # If hand is in center area, consider it targeting
        if abs(x - center_x) < 100 and abs(y - center_y) < 100:
            if not data.target_locked and np.random.random() < 0.05:
                data.target_locked = True
                data.target_distance = np.random.uniform(5, 15)
        
        # Update speed based on number of hands
        if data.num_hands >= 2:
            data.airspeed += 0.5
        
    return data

def detect_hands(frame, model):
    """Detect hands in frame using YOLO model"""
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
            
            # Determine hand gesture (simplified)
            hand_width = x2 - x1
            hand_height = y2 - y1
            aspect_ratio = hand_width / hand_height if hand_height > 0 else 1
            
            if aspect_ratio > 1.5:
                data.hand_gesture = "Open Palm"
            elif aspect_ratio < 0.6:
                data.hand_gesture = "Pointing"
            else:
                data.hand_gesture = "Closed"
            
            # Draw boxes on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{data.hand_gesture} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # Analyze hand movement effects on flight parameters
    data = analyze_hand_movement(data)
    
    return frame, data

def draw_enhanced_hud(screen, frame_surface, data):
    """Draw all HUD elements with enhanced visuals"""
    # First blit the camera frame
    screen.blit(frame_surface, (0, 0))
    
    # Draw main HUD elements
    draw_airspeed_indicator(screen, data.airspeed)
    draw_altitude_indicator(screen, data.altitude)
    draw_heading_indicator(screen, data.heading)
    draw_horizon(screen, data.pitch, data.roll)
    draw_flight_path_vector(screen, data.pitch, data.roll)
    draw_status_indicators(screen, data)  # Fixed: Pass the entire data object, not just g_force
    draw_target_info(screen, data)
    
    # Draw hand detection visualization if hand is detected
    if data.hand_detected:
        draw_hand_detection_info(screen, data)
    
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

def run_analyzer(analyzer, hud_data_ref):
    """Run the analysis window in a separate thread"""
    # Start Tkinter in this thread
    analyzer.root.after(100, update_analyzer_loop, analyzer, hud_data_ref)
    analyzer.root.mainloop()

def update_analyzer_loop(analyzer, hud_data_ref):
    """Update analyzer data periodically through Tkinter's event loop"""
    try:
        analyzer.update(hud_data_ref[0])
    except Exception as e:
        print(f"Error updating analytics window: {e}")
    
    # Schedule next update
    analyzer.root.after(50, update_analyzer_loop, analyzer, hud_data_ref)

def main():
    """Main function to run the camera-based HUD with hand detection"""
    # Load YOLO model for hand detection
    print("Loading YOLO model for hand detection...")
    model = YOLO("yolov8n.pt")  # Use yolov8 nano model
    
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
    pygame.display.set_caption("Fighter Jet HUD with Hand Detection")
    
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
    
    print("HUD system ready. Press ESC to exit.")
    
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
        
        # Run hand detection on frame
        frame, hud_data = detect_hands(frame, model)
        
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
    pygame.quit()
    print("HUD system terminated.")

if __name__ == "__main__":
    main()