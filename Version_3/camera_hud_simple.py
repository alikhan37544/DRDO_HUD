import cv2
import pygame
import numpy as np
import math
import random
import time
from datetime import datetime
import csv
import os

# Initialize pygame
pygame.init()

# Display parameters
WIDTH, HEIGHT = 1280, 720
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 160, 255)
WHITE = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 140, 0)  # Orange for highlights

# Font setup
font_small = pygame.font.SysFont('Arial', 18)
font_medium = pygame.font.SysFont('Arial', 24)
font_large = pygame.font.SysFont('Arial', 32)

# Create display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Aircraft HUD Camera System")

# Initialize camera
camera = cv2.VideoCapture(0)  # Use 0 for default camera
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution to match display
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# HUD Parameters
class HUDData:
    def __init__(self):
        # Basic flight data
        self.altitude = 5000  # feet
        self.airspeed = 320   # knots
        self.heading = 45     # degrees
        self.pitch = 0        # degrees
        self.roll = 0         # degrees
        
        # Advanced flight data
        self.g_force = 1.0    # G
        self.aoa = 2.5        # Angle of attack (degrees)
        self.mach = 0.75      # Mach number
        self.fuel = 70        # Percentage
        self.vertical_speed = 0  # feet per minute
        
        # Navigation and targeting
        self.weapon_status = "READY"
        self.target_distance = 12.5  # km
        self.target_locked = False
        self.waypoint_bearing = 85   # degrees
        self.waypoint_distance = 45  # km
        self.radar_contacts = 2      # Number of contacts
        
        # System status
        self.engine_temp = 85        # percentage
        self.system_status = "NOMINAL"
        self.threat_level = "LOW"
        self.ecm_status = "STANDBY"
        
        # Symbols (represented as integers for simplicity)
        self.symbol_attack = 0       # 0-3 different attack patterns
        self.symbol_defense = 1      # 0-3 different defense patterns
        self.symbol_navigation = 2   # 0-3 different navigation aids
        
        # Time
        self.time = datetime.now()
        
        # Previous values for change detection
        self.previous_values = self.__dict__.copy()

    def detect_changes(self):
        """Detect which values have changed since last check"""
        changes = {}
        for key, value in self.__dict__.items():
            if key != 'previous_values' and key != 'time':
                if key in self.previous_values and self.previous_values[key] != value:
                    # Calculate the delta
                    if isinstance(value, (int, float)):
                        delta = value - self.previous_values[key]
                        changes[key] = (value, delta)
                    else:
                        changes[key] = (value, "changed")
                        
        # Update previous values
        self.previous_values = self.__dict__.copy()
        return changes

# CSV logging setup
def setup_csv_logging():
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hud_data_{timestamp}.csv"
    
    # Create CSV file and write header
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'altitude', 'airspeed', 'heading', 'pitch', 'roll', 
            'g_force', 'aoa', 'mach', 'fuel', 'vertical_speed', 'weapon_status',
            'target_distance', 'target_locked', 'waypoint_bearing', 
            'waypoint_distance', 'radar_contacts', 'engine_temp', 
            'system_status', 'threat_level', 'ecm_status',
            'symbol_attack', 'symbol_defense', 'symbol_navigation'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def log_hud_data(filename, data):
    """Log HUD data to CSV file"""
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'timestamp', 'altitude', 'airspeed', 'heading', 'pitch', 'roll', 
            'g_force', 'aoa', 'mach', 'fuel', 'vertical_speed', 'weapon_status',
            'target_distance', 'target_locked', 'waypoint_bearing', 
            'waypoint_distance', 'radar_contacts', 'engine_temp', 
            'system_status', 'threat_level', 'ecm_status',
            'symbol_attack', 'symbol_defense', 'symbol_navigation'
        ])
        
        # Prepare data for logging
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'altitude': data.altitude,
            'airspeed': data.airspeed,
            'heading': data.heading,
            'pitch': data.pitch,
            'roll': data.roll,
            'g_force': data.g_force,
            'aoa': data.aoa,
            'mach': data.mach,
            'fuel': data.fuel,
            'vertical_speed': data.vertical_speed,
            'weapon_status': data.weapon_status,
            'target_distance': data.target_distance,
            'target_locked': data.target_locked,
            'waypoint_bearing': data.waypoint_bearing,
            'waypoint_distance': data.waypoint_distance,
            'radar_contacts': data.radar_contacts,
            'engine_temp': data.engine_temp,
            'system_status': data.system_status,
            'threat_level': data.threat_level,
            'ecm_status': data.ecm_status,
            'symbol_attack': data.symbol_attack,
            'symbol_defense': data.symbol_defense,
            'symbol_navigation': data.symbol_navigation
        }
        
        writer.writerow(log_entry)

# Convert OpenCV image to Pygame surface
def convert_opencv_to_pygame(opencv_image):
    # Convert from BGR to RGB color format
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # Convert to Pygame surface
    pygame_surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
    return pygame_surface

# HUD drawing functions
def draw_airspeed_indicator(screen, airspeed):
    pygame.draw.rect(screen, (0, 0, 0, 180), (50, HEIGHT//2-150, 100, 300))
    pygame.draw.rect(screen, GREEN, (50, HEIGHT//2-150, 100, 300), 2)
    
    text = font_large.render(f"{int(airspeed)}", True, GREEN)
    screen.blit(text, (75, HEIGHT//2-10))
    
    text = font_small.render("KIAS", True, GREEN)
    screen.blit(text, (75, HEIGHT//2+30))
    
    # Speed ticks
    for i in range(-5, 6):
        if i == 0:
            continue
        tick_speed = int(airspeed) + i * 20
        y_pos = HEIGHT//2 - i * 25
        pygame.draw.line(screen, GREEN, (50, y_pos), (65, y_pos), 2)
        text = font_small.render(f"{tick_speed}", True, GREEN)
        screen.blit(text, (70, y_pos-10))

def draw_altitude_indicator(screen, altitude):
    pygame.draw.rect(screen, (0, 0, 0, 180), (WIDTH-150, HEIGHT//2-150, 100, 300))
    pygame.draw.rect(screen, GREEN, (WIDTH-150, HEIGHT//2-150, 100, 300), 2)
    
    text = font_large.render(f"{int(altitude)}", True, GREEN)
    screen.blit(text, (WIDTH-130, HEIGHT//2-10))
    
    text = font_small.render("ALT", True, GREEN)
    screen.blit(text, (WIDTH-125, HEIGHT//2+30))
    
    # Altitude ticks
    for i in range(-5, 6):
        if i == 0:
            continue
        tick_alt = int(altitude) + i * 500
        y_pos = HEIGHT//2 - i * 25
        pygame.draw.line(screen, GREEN, (WIDTH-150, y_pos), (WIDTH-135, y_pos), 2)
        text = font_small.render(f"{tick_alt}", True, GREEN)
        screen.blit(text, (WIDTH-130, y_pos-10))

def draw_heading_indicator(screen, heading):
    center_x = WIDTH // 2
    
    # Draw compass arc
    pygame.draw.rect(screen, (0, 0, 0, 180), (center_x-150, 30, 300, 50))
    pygame.draw.rect(screen, GREEN, (center_x-150, 30, 300, 50), 2)
    pygame.draw.line(screen, GREEN, (center_x, 30), (center_x, 50), 2)
    
    # Draw heading value
    text = font_medium.render(f"{int(heading):03d}°", True, GREEN)
    screen.blit(text, (center_x-20, 35))
    
    # Draw heading ticks
    for i in range(-4, 5):
        tick_heading = (heading + i * 10) % 360
        x_pos = center_x + i * 30
        if x_pos >= center_x-150 and x_pos <= center_x+150:
            pygame.draw.line(screen, GREEN, (x_pos, 70), (x_pos, 80), 2)
            text = font_small.render(f"{int(tick_heading):03d}", True, GREEN)
            screen.blit(text, (x_pos-10, 80))

def draw_horizon(screen, pitch, roll):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    horizon_width = 600
    
    # Calculate horizon line based on pitch and roll
    roll_rad = math.radians(roll)
    pitch_offset = pitch * 5  # 5 pixels per degree of pitch
    
    # Calculate horizon endpoints
    x1 = center_x - (horizon_width/2) * math.cos(roll_rad)
    y1 = center_y + (horizon_width/2) * math.sin(roll_rad) - pitch_offset
    x2 = center_x + (horizon_width/2) * math.cos(roll_rad)
    y2 = center_y - (horizon_width/2) * math.sin(roll_rad) - pitch_offset
    
    # Draw artificial horizon
    pygame.draw.line(screen, GREEN, (int(x1), int(y1)), (int(x2), int(y2)), 2)
    
    # Draw pitch ladder
    for p in range(-20, 21, 5):
        if p == 0:
            continue
        
        ladder_width = 100 if p % 10 == 0 else 50
        pitch_pixel = p * 5
        
        x1 = center_x - (ladder_width/2) * math.cos(roll_rad)
        y1 = center_y + (ladder_width/2) * math.sin(roll_rad) - pitch_offset - pitch_pixel
        x2 = center_x + (ladder_width/2) * math.cos(roll_rad)
        y2 = center_y - (ladder_width/2) * math.sin(roll_rad) - pitch_offset - pitch_pixel
        
        if -200 < y1 < HEIGHT+200 and -200 < y2 < HEIGHT+200:
            pygame.draw.line(screen, GREEN, (int(x1), int(y1)), (int(x2), int(y2)), 2)
            text = font_small.render(f"{abs(p)}", True, GREEN)
            screen.blit(text, (int(x2)+10, int(y2)-10))

def draw_flight_path_vector(screen, pitch, roll):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    
    # Draw flight path vector (FPV) symbol
    pygame.draw.circle(screen, GREEN, (center_x, center_y), 10, 2)
    pygame.draw.line(screen, GREEN, (center_x-20, center_y), (center_x-10, center_y), 2)
    pygame.draw.line(screen, GREEN, (center_x+10, center_y), (center_x+20, center_y), 2)
    pygame.draw.line(screen, GREEN, (center_x, center_y-20), (center_x, center_y-10), 2)

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
    
    # Vertical speed
    vs_color = GREEN
    if abs(data.vertical_speed) > 1000:
        vs_color = YELLOW
    text = font_medium.render(f"VS: {data.vertical_speed:+}", True, vs_color)
    screen.blit(text, (50, HEIGHT-140))
    
    # Fuel
    fuel_color = GREEN if data.fuel > 20 else RED
    text = font_medium.render(f"FUEL: {data.fuel}%", True, fuel_color)
    screen.blit(text, (WIDTH-200, HEIGHT-50))
    
    # Engine temp
    temp_color = GREEN
    if data.engine_temp > 90:
        temp_color = RED
    elif data.engine_temp > 80:
        temp_color = YELLOW
    text = font_medium.render(f"ENG: {data.engine_temp}%", True, temp_color)
    screen.blit(text, (WIDTH-200, HEIGHT-80))
    
    # Weapon status
    text = font_medium.render(f"WPN: {data.weapon_status}", True, GREEN)
    screen.blit(text, (WIDTH-200, HEIGHT-110))
    
    # System status
    status_color = GREEN
    if data.system_status != "NOMINAL":
        status_color = YELLOW
    text = font_medium.render(f"SYS: {data.system_status}", True, status_color)
    screen.blit(text, (WIDTH-200, HEIGHT-140))
    
    # Threat level
    threat_color = GREEN
    if data.threat_level == "HIGH":
        threat_color = RED
    elif data.threat_level == "MEDIUM":
        threat_color = YELLOW
    text = font_medium.render(f"THREAT: {data.threat_level}", True, threat_color)
    screen.blit(text, (WIDTH-200, HEIGHT-170))
    
    # ECM status
    ecm_color = BLUE if data.ecm_status == "ACTIVE" else GREEN
    text = font_medium.render(f"ECM: {data.ecm_status}", True, ecm_color)
    screen.blit(text, (WIDTH-200, HEIGHT-200))
    
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
    
    # Radar contacts
    text = font_medium.render(f"RADAR: {data.radar_contacts} CONTACTS", True, GREEN)
    screen.blit(text, (WIDTH//2-100, 150))
    
    # Waypoint
    text = font_medium.render(f"WPT: {data.waypoint_distance:.1f}KM / {int(data.waypoint_bearing):03d}°", True, GREEN)
    screen.blit(text, (WIDTH//2+50, 100))

def draw_symbols(screen, data):
    # Attack symbol
    attack_symbols = [
        lambda x, y: pygame.draw.rect(screen, RED, (x-15, y-15, 30, 30), 2),  # Square
        lambda x, y: pygame.draw.polygon(screen, RED, [(x, y-20), (x+20, y+10), (x-20, y+10)], 2),  # Triangle
        lambda x, y: pygame.draw.circle(screen, RED, (x, y), 15, 2),  # Circle
        lambda x, y: pygame.draw.polygon(screen, RED, [(x-15, y), (x, y-15), (x+15, y), (x, y+15)], 2)  # Diamond
    ]
    
    attack_symbols[data.symbol_attack](100, 150)
    text = font_small.render("ATTACK", True, RED)
    screen.blit(text, (75, 175))
    
    # Defense symbol
    defense_symbols = [
        lambda x, y: pygame.draw.rect(screen, BLUE, (x-15, y-15, 30, 30), 2),
        lambda x, y: pygame.draw.circle(screen, BLUE, (x, y), 15, 2),
        lambda x, y: pygame.draw.polygon(screen, BLUE, [(x-15, y-15), (x+15, y-15), (x, y+15)], 2),
        lambda x, y: pygame.draw.lines(screen, BLUE, True, [(x-15, y-15), (x+15, y-15), (x+15, y+15), (x-15, y+15)], 2)
    ]
    
    defense_symbols[data.symbol_defense](100, 220)
    text = font_small.render("DEFENSE", True, BLUE)
    screen.blit(text, (75, 245))
    
    # Navigation symbol
    nav_symbols = [
        lambda x, y: pygame.draw.polygon(screen, GREEN, [(x, y-15), (x+10, y+10), (x-10, y+10)], 2),
        lambda x, y: pygame.draw.circle(screen, GREEN, (x, y), 12, 2),
        lambda x, y: pygame.draw.lines(screen, GREEN, True, [(x, y-15), (x+15, y), (x, y+15), (x-15, y)], 2),
        lambda x, y: pygame.draw.polygon(screen, GREEN, [(x-10, y-10), (x+10, y-10), (x+10, y+10), (x-10, y+10)], 2)
    ]
    
    nav_symbols[data.symbol_navigation](100, 290)
    text = font_small.render("NAV", True, GREEN)
    screen.blit(text, (90, 315))

def draw_highlight_panel(screen, changes):
    if not changes:
        return
        
    # Draw panel background
    panel_width = 300
    panel_height = min(len(changes) * 30 + 40, 200)
    panel_x = WIDTH - panel_width - 20
    panel_y = 20
    
    pygame.draw.rect(screen, (0, 0, 0, 200), (panel_x, panel_y, panel_width, panel_height))
    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (panel_x, panel_y, panel_width, panel_height), 2)
    
    # Draw panel title
    text = font_medium.render("CHANGES DETECTED", True, HIGHLIGHT_COLOR)
    screen.blit(text, (panel_x + 10, panel_y + 10))
    
    # Draw changes
    y_offset = panel_y + 40
    for key, (value, delta) in list(changes.items())[:5]:  # Show at most 5 changes
        if isinstance(delta, (int, float)):
            delta_str = f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
            text = font_small.render(f"{key}: {value} ({delta_str})", True, HIGHLIGHT_COLOR)
        else:
            text = font_small.render(f"{key}: {value}", True, HIGHLIGHT_COLOR)
        screen.blit(text, (panel_x + 10, y_offset))
        y_offset += 30
        
    # If there are more changes than can fit
    if len(changes) > 5:
        text = font_small.render(f"...and {len(changes) - 5} more", True, HIGHLIGHT_COLOR)
        screen.blit(text, (panel_x + 10, y_offset))

def update_hud_values(data):
    # Update with random realistic changes
    data.altitude += random.uniform(-20, 20)
    data.airspeed += random.uniform(-2, 2)
    data.heading = (data.heading + random.uniform(-0.5, 0.5)) % 360
    data.pitch = max(-30, min(30, data.pitch + random.uniform(-0.3, 0.3)))
    data.roll = max(-45, min(45, data.roll + random.uniform(-0.5, 0.5)))
    data.g_force = max(0.1, min(9.0, data.g_force + random.uniform(-0.05, 0.05)))
    data.aoa = max(-10, min(20, data.aoa + random.uniform(-0.1, 0.1)))
    data.mach = max(0.5, min(1.2, data.mach + random.uniform(-0.01, 0.01)))
    data.fuel = max(0, min(100, data.fuel - random.uniform(0.01, 0.05)))
    data.vertical_speed = max(-2000, min(2000, data.vertical_speed + random.uniform(-50, 50)))
    data.engine_temp = max(70, min(100, data.engine_temp + random.uniform(-0.5, 0.5)))
    
    # System status changes
    if random.random() < 0.01:
        statuses = ["NOMINAL", "CHECK", "WARNING", "CAUTION"]
        data.system_status = random.choice(statuses)
    
    # Threat level changes
    if random.random() < 0.01:
        threat_levels = ["LOW", "MEDIUM", "HIGH"]
        data.threat_level = random.choice(threat_levels)
    
    # ECM status changes
    if random.random() < 0.01:
        ecm_statuses = ["STANDBY", "ACTIVE", "PASSIVE"]
        data.ecm_status = random.choice(ecm_statuses)
    
    # Target tracking
    if random.random() < 0.01:
        data.target_locked = not data.target_locked
    
    if data.target_locked:
        data.target_distance = max(0.5, min(25, data.target_distance + random.uniform(-0.2, 0.2)))
    
    # Radar contacts
    if random.random() < 0.05:
        data.radar_contacts = max(0, min(10, data.radar_contacts + random.choice([-1, 0, 1])))
    
    # Waypoint updates
    data.waypoint_distance = max(0, data.waypoint_distance - random.uniform(0.05, 0.1))
    data.waypoint_bearing = (data.waypoint_bearing + random.uniform(-0.2, 0.2)) % 360
    
    # Symbol changes (less frequent)
    if random.random() < 0.03:
        data.symbol_attack = random.randint(0, 3)
    if random.random() < 0.03:
        data.symbol_defense = random.randint(0, 3)
    if random.random() < 0.03:
        data.symbol_navigation = random.randint(0, 3)
    
    # Update time
    data.time = datetime.now()
    
    return data

def main():
    hud_data = HUDData()
    clock = pygame.time.Clock()
    running = True
    
    # Set up CSV logging
    csv_filename = setup_csv_logging()
    
    # Track changes for highlighting
    current_changes = {}
    change_display_time = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                # Add some interactive controls
                if event.key == pygame.K_UP:
                    hud_data.pitch += 5
                if event.key == pygame.K_DOWN:
                    hud_data.pitch -= 5
                if event.key == pygame.K_LEFT:
                    hud_data.roll -= 5
                if event.key == pygame.K_RIGHT:
                    hud_data.roll += 5
                if event.key == pygame.K_a:
                    hud_data.airspeed += 10
                if event.key == pygame.K_z:
                    hud_data.airspeed -= 10
                if event.key == pygame.K_s:
                    hud_data.altitude += 100
                if event.key == pygame.K_x:
                    hud_data.altitude -= 100
                if event.key == pygame.K_t:
                    hud_data.target_locked = not hud_data.target_locked
        
        # Capture camera frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read from camera.")
            break
            
        # Mirror image (so it acts like a mirror)
        frame = cv2.flip(frame, 1)
        
        # Convert camera frame to pygame surface
        camera_surface = convert_opencv_to_pygame(frame)
        
        # Update HUD values
        hud_data = update_hud_values(hud_data)
        
        # Detect changes
        changes = hud_data.detect_changes()
        if changes:
            current_changes = changes
            change_display_time = time.time() + 3  # Display for 3 seconds
        
        # Clear expired changes display
        if time.time() > change_display_time:
            current_changes = {}
        
        # Draw the camera feed
        screen.blit(camera_surface, (0, 0))
        
        # Apply a slight darkening overlay to make HUD more visible
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 50))  # Semi-transparent black
        screen.blit(overlay, (0, 0))
        
        # Draw HUD elements
        draw_airspeed_indicator(screen, hud_data.airspeed)
        draw_altitude_indicator(screen, hud_data.altitude)
        draw_heading_indicator(screen, hud_data.heading)
        draw_horizon(screen, hud_data.pitch, hud_data.roll)
        draw_flight_path_vector(screen, hud_data.pitch, hud_data.roll)
        draw_status_indicators(screen, hud_data)
        draw_target_info(screen, hud_data)
        draw_symbols(screen, hud_data)
        
        # Draw change highlights
        if current_changes:
            draw_highlight_panel(screen, current_changes)
        
        # Log data to CSV
        log_hud_data(csv_filename, hud_data)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    # Clean up
    camera.release()
    pygame.quit()
    print(f"HUD session ended. Data saved to {csv_filename}")

if __name__ == "__main__":
    main()