import pygame
import numpy as np
import math
import random
import time
from datetime import datetime

# Initialize pygame
pygame.init()

# Display parameters
WIDTH, HEIGHT = 1280, 720
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# Font setup
font_small = pygame.font.SysFont('Arial', 18)
font_medium = pygame.font.SysFont('Arial', 24)
font_large = pygame.font.SysFont('Arial', 32)

# Create display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Aircraft HUD Simulation")

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

# HUD drawing functions
def draw_airspeed_indicator(screen, airspeed):
    pygame.draw.rect(screen, BLACK, (50, HEIGHT//2-150, 100, 300))
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
    pygame.draw.rect(screen, BLACK, (WIDTH-150, HEIGHT//2-150, 100, 300))
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
    pygame.draw.rect(screen, BLACK, (center_x-150, 30, 300, 50))
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
    text = font_medium.render(f"WPT: {data.waypoint_distance:.1f}KM / {data.waypoint_bearing:03d}°", True, GREEN)
    screen.blit(text, (WIDTH//2+50, 100))

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
    
    if random.random() < 0.01:
        data.target_locked = not data.target_locked
    
    if data.target_locked:
        data.target_distance = max(0.5, min(25, data.target_distance + random.uniform(-0.2, 0.2)))
    
    data.waypoint_distance = max(0, data.waypoint_distance - random.uniform(0.05, 0.1))
    data.waypoint_bearing = (data.waypoint_bearing + random.uniform(-0.2, 0.2)) % 360
    data.time = datetime.now()
    
    return data

def main():
    hud_data = HUDData()
    clock = pygame.time.Clock()
    running = True
    
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
        
        # Update HUD values
        hud_data = update_hud_values(hud_data)
        
        # Draw HUD
        screen.fill(BLACK)
        
        # Draw HUD elements
        draw_airspeed_indicator(screen, hud_data.airspeed)
        draw_altitude_indicator(screen, hud_data.altitude)
        draw_heading_indicator(screen, hud_data.heading)
        draw_horizon(screen, hud_data.pitch, hud_data.roll)
        draw_flight_path_vector(screen, hud_data.pitch, hud_data.roll)
        draw_status_indicators(screen, hud_data)
        draw_target_info(screen, hud_data)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()