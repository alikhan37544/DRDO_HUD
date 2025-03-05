import pygame
import random
import math
import csv
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
BG_COLOR = (0, 0, 0)
HUD_COLOR = (0, 255, 120)
WARNING_COLOR = (255, 200, 0)
DANGER_COLOR = (255, 0, 0)
HIGHLIGHT_COLOR = (100, 255, 150)
CAMERA_IMG_PATH = "camera.jpg"  # Replace with your actual camera image path

# Symbols for the HUD
SYMBOLS = ['◆', '■', '●', '▲', '★', '✦', '⚠', '◎', '◈', '⬢', '⬡', '△', '▣', '◇', '◉']

# Shape types and colors
SHAPES = [
    {"type": "circle", "color": (52, 152, 219)},    # Blue
    {"type": "square", "color": (231, 76, 60)},     # Red
    {"type": "triangle", "color": (46, 204, 113)},  # Green
    {"type": "diamond", "color": (243, 156, 18)},   # Orange
    {"type": "hexagon", "color": (155, 89, 182)}    # Purple
]

class ChangeRecord:
    """Class to track and record changes in HUD data"""
    def __init__(self):
        self.changes = []
        self.csv_data = []
        
    def add_change(self, change_type: str, old_value: Any, new_value: Any, **kwargs) -> Dict:
        """Add a change to the records and return the change data"""
        timestamp = datetime.now().isoformat()
        
        change_data = {
            "time": timestamp,
            "type": change_type,
            "oldValue": old_value,
            "newValue": new_value,
            **kwargs
        }
        
        # Calculate difference for numeric values
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            diff = new_value - old_value
            change_data["diff"] = diff
            
            # Calculate percentage change if old value is not zero
            if old_value != 0:
                diff_percent = (diff / old_value) * 100
                change_data["diffPercent"] = round(diff_percent, 1)
        
        # Add to both live changes display and CSV data
        self.changes.insert(0, change_data)
        if len(self.changes) > 15:  # Keep only last 15 changes for display
            self.changes = self.changes[:15]
            
        # Add to CSV data (keep all changes)
        self.csv_data.append(change_data)
        
        return change_data
    
    def export_to_csv(self) -> str:
        """Export all recorded changes to a CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hud_data_{timestamp}.csv"
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(self.csv_data)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        return filename


class Symbol:
    """Class representing a symbol on the HUD"""
    def __init__(self, symbol_id: int, symbol: str, position: str, status: str = "normal"):
        self.id = symbol_id
        self.symbol = symbol
        self.position = position
        self.status = status  # "normal" or "warning"
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for comparison"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "position": self.position,
            "status": self.status
        }


class Shape:
    """Class representing a tactical shape on the HUD"""
    def __init__(self, shape_id: int, shape_type: str, color: Tuple[int, int, int], 
                 x: int, y: int, size: int):
        self.id = shape_id
        self.type = shape_type
        self.color = color
        self.x = x
        self.y = y
        self.size = size
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for comparison"""
        return {
            "id": self.id,
            "type": self.type,
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "size": self.size
        }
        
    def draw(self, surface):
        """Draw the shape on the given surface"""
        if self.type == "circle":
            pygame.draw.circle(surface, self.color, (self.x, self.y), self.size, 2)
        elif self.type == "square":
            rect = pygame.Rect(self.x - self.size, self.y - self.size, 
                               self.size * 2, self.size * 2)
            pygame.draw.rect(surface, self.color, rect, 2)
        elif self.type == "triangle":
            points = [
                (self.x, self.y - self.size),
                (self.x - self.size, self.y + self.size),
                (self.x + self.size, self.y + self.size)
            ]
            pygame.draw.polygon(surface, self.color, points, 2)
        elif self.type == "diamond":
            points = [
                (self.x, self.y - self.size),
                (self.x + self.size, self.y),
                (self.x, self.y + self.size),
                (self.x - self.size, self.y)
            ]
            pygame.draw.polygon(surface, self.color, points, 2)
        elif self.type == "hexagon":
            points = []
            for i in range(6):
                angle = math.radians(60 * i)
                points.append((
                    self.x + self.size * math.cos(angle),
                    self.y + self.size * math.sin(angle)
                ))
            pygame.draw.polygon(surface, self.color, points, 2)


class HUD:
    """Main HUD class managing all elements and interactions"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dynamic HUD System")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Load camera background (or create a placeholder)
        try:
            self.camera_img = pygame.image.load(CAMERA_IMG_PATH)
            self.camera_img = pygame.transform.scale(self.camera_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except:
            # Create a placeholder gradient background
            self.camera_img = self.create_placeholder_background()
        
        # Load fonts
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 18)
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_tiny = pygame.font.SysFont("monospace", 12)
        self.font_symbols = pygame.font.SysFont("segoe ui symbol", 24)  # Font with good symbol support
        
        # HUD Data
        self.altitude = 10000
        self.speed = 450
        self.heading = 120
        self.fuel_level = 87
        self.temperature = 23
        
        # Symbols and shapes
        self.symbols = [
            Symbol(1, SYMBOLS[0], "top-left", "normal"),
            Symbol(2, SYMBOLS[3], "top-right", "warning"),
            Symbol(3, SYMBOLS[7], "bottom-left", "normal")
        ]
        
        self.shapes = [
            Shape(1, "circle", (52, 152, 219), 300, 400, 15),
            Shape(2, "square", (231, 76, 60), 800, 300, 20),
            Shape(3, "triangle", (46, 204, 113), 500, 600, 25)
        ]
        
        # Change tracking
        self.change_tracker = ChangeRecord()
        
        # For comparison to detect changes
        self.previous_values = {
            "altitude": self.altitude,
            "speed": self.speed,
            "heading": self.heading,
            "fuel_level": self.fuel_level,
            "temperature": self.temperature,
            "symbols": [s.to_dict() for s in self.symbols],
            "shapes": [s.to_dict() for s in self.shapes]
        }
        
        # For managing change highlights
        self.highlights = []
        
        # For timing updates
        self.last_update_time = pygame.time.get_ticks()
        self.update_interval = 2000  # milliseconds
        
    def create_placeholder_background(self) -> pygame.Surface:
        """Create a placeholder background if no camera image is available"""
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Create a gradient from dark blue to black
        for y in range(SCREEN_HEIGHT):
            # Calculate color based on y position
            color_value = max(0, 30 - int(y / SCREEN_HEIGHT * 30))
            color = (0, 0, color_value)
            pygame.draw.line(surf, color, (0, y), (SCREEN_WIDTH, y))
            
        # Add some "stars"
        for _ in range(200):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            brightness = random.randint(100, 255)
            size = random.randint(1, 3)
            pygame.draw.circle(surf, (brightness, brightness, brightness), (x, y), size)
            
        return surf
            
    def update_hud_data(self):
        """Randomly update HUD data periodically"""
        current_time = pygame.time.get_ticks()
        
        # Update every few seconds
        if current_time - self.last_update_time > self.update_interval:
            self.last_update_time = current_time
            
            # Store previous values for comparison
            self.previous_values = {
                "altitude": self.altitude,
                "speed": self.speed,
                "heading": self.heading,
                "fuel_level": self.fuel_level,
                "temperature": self.temperature,
                "symbols": [s.to_dict() for s in self.symbols],
                "shapes": [s.to_dict() for s in self.shapes]
            }
            
            # Randomly update values
            if random.random() > 0.7:
                self.altitude = int(self.altitude + (random.random() * 200 - 100))
                self._check_and_record_change("altitude", self.previous_values["altitude"], self.altitude)
                
            if random.random() > 0.7:
                self.speed = int(self.speed + (random.random() * 20 - 10))
                self._check_and_record_change("speed", self.previous_values["speed"], self.speed)
                
            if random.random() > 0.7:
                self.heading = (self.heading + int(random.random() * 10 - 5)) % 360
                self._check_and_record_change("heading", self.previous_values["heading"], self.heading)
                
            if random.random() > 0.7:
                self.fuel_level = max(0, min(100, self.fuel_level + (random.random() * 4 - 2)))
                self._check_and_record_change("fuel_level", self.previous_values["fuel_level"], self.fuel_level)
                
            if random.random() > 0.7:
                self.temperature = int(self.temperature + (random.random() * 2 - 1))
                self._check_and_record_change("temperature", self.previous_values["temperature"], self.temperature)
                
            # Random symbol changes
            if random.random() > 0.8:
                rand_idx = random.randint(0, len(self.symbols) - 1)
                old_symbol = self.symbols[rand_idx].to_dict()
                
                # Change symbol
                self.symbols[rand_idx].symbol = random.choice(SYMBOLS)
                self._check_and_record_symbol_change(old_symbol, self.symbols[rand_idx].to_dict())
                
                # Maybe change status
                if random.random() > 0.5:
                    old_status = self.symbols[rand_idx].status
                    self.symbols[rand_idx].status = "warning" if old_status == "normal" else "normal"
                    self._check_and_record_symbol_status_change(old_symbol, self.symbols[rand_idx].to_dict())
                    
            # Random shape changes
            if random.random() > 0.8:
                rand_idx = random.randint(0, len(self.shapes) - 1)
                old_shape = self.shapes[rand_idx].to_dict()
                
                random_shape = random.choice(SHAPES)
                self.shapes[rand_idx].type = random_shape["type"]
                self.shapes[rand_idx].color = random_shape["color"]
                self.shapes[rand_idx].x = random.randint(200, SCREEN_WIDTH - 200)
                self.shapes[rand_idx].y = random.randint(200, SCREEN_HEIGHT - 200)
                self.shapes[rand_idx].size = random.randint(15, 30)
                
                self._check_and_record_shape_change(old_shape, self.shapes[rand_idx].to_dict())
    
    def _check_and_record_change(self, change_type, old_value, new_value):
        """Check if value has changed and record it"""
        if old_value != new_value:
            change = self.change_tracker.add_change(change_type, old_value, new_value)
            self.highlights.append({"type": change_type, "time": pygame.time.get_ticks(), "data": change})
            
    def _check_and_record_symbol_change(self, old_symbol, new_symbol):
        """Check and record symbol changes"""
        if old_symbol["symbol"] != new_symbol["symbol"]:
            change = self.change_tracker.add_change(
                "symbol", 
                old_symbol["symbol"],
                new_symbol["symbol"],
                position=new_symbol["position"],
                id=new_symbol["id"]
            )
            self.highlights.append({"type": "symbol", "time": pygame.time.get_ticks(), "data": change})
            
    def _check_and_record_symbol_status_change(self, old_symbol, new_symbol):
        """Check and record symbol status changes"""
        if old_symbol["status"] != new_symbol["status"]:
            change = self.change_tracker.add_change(
                "symbolStatus", 
                old_symbol["status"],
                new_symbol["status"],
                position=new_symbol["position"],
                id=new_symbol["id"]
            )
            self.highlights.append({"type": "symbolStatus", "time": pygame.time.get_ticks(), "data": change})
            
    def _check_and_record_shape_change(self, old_shape, new_shape):
        """Check and record shape changes"""
        if (old_shape["type"] != new_shape["type"] or
            old_shape["color"] != new_shape["color"] or
            old_shape["size"] != new_shape["size"] or
            old_shape["x"] != new_shape["x"] or
            old_shape["y"] != new_shape["y"]):
            
            change = self.change_tracker.add_change(
                "shape",
                f"{old_shape['type']}",
                f"{new_shape['type']}",
                id=new_shape["id"],
                oldColor=old_shape["color"],
                newColor=new_shape["color"],
                oldSize=old_shape["size"],
                newSize=new_shape["size"],
                oldPosition=f"x:{old_shape['x']},y:{old_shape['y']}",
                newPosition=f"x:{new_shape['x']},y:{new_shape['y']}"
            )
            self.highlights.append({"type": "shape", "time": pygame.time.get_ticks(), "data": change})
    
    def draw_hud(self):
        """Draw all HUD elements"""
        # Draw camera feed background with reduced opacity
        self.screen.blit(self.camera_img, (0, 0))
        
        # Create a semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Top bar
        pygame.draw.line(self.screen, HUD_COLOR, (10, 50), (SCREEN_WIDTH - 10, 50), 2)
        
        # Draw main sections
        self.draw_top_bar()
        self.draw_left_panel()
        self.draw_center_view()
        self.draw_right_panel()
        self.draw_bottom_bar()
        
    def draw_top_bar(self):
        """Draw the top information bar"""
        # Heading
        hdg_text = self.font_large.render(f"HDG: {self.heading}°", True, HUD_COLOR)
        self.screen.blit(hdg_text, (20, 15))
        
        # Altitude
        alt_text = self.font_large.render(f"ALT: {self.altitude:,} ft", True, HUD_COLOR)
        self.screen.blit(alt_text, (SCREEN_WIDTH // 2 - alt_text.get_width() // 2, 15))
        
        # Speed
        spd_text = self.font_large.render(f"SPD: {self.speed} kts", True, HUD_COLOR)
        self.screen.blit(spd_text, (SCREEN_WIDTH - 20 - spd_text.get_width(), 15))
        
        # Temperature
        temp_text = self.font_large.render(f"{self.temperature}°C", True, HUD_COLOR)
        self.screen.blit(temp_text, (SCREEN_WIDTH - 150, 15))
        
    def draw_left_panel(self):
        """Draw the left information panel"""
        # Panel area
        panel_rect = pygame.Rect(10, 60, 300, SCREEN_HEIGHT - 110)
        pygame.draw.rect(self.screen, HUD_COLOR, panel_rect, 1)
        
        # System status
        status_text = self.font_medium.render("SYSTEM STATUS", True, HUD_COLOR)
        self.screen.blit(status_text, (20, 70))
        
        # Fuel gauge
        fuel_label = self.font_small.render("FUEL LEVEL", True, HUD_COLOR)
        self.screen.blit(fuel_label, (20, 100))
        
        fuel_value_color = DANGER_COLOR if self.fuel_level < 20 else HUD_COLOR
        fuel_value = self.font_small.render(f"{self.fuel_level:.1f}%", True, fuel_value_color)
        self.screen.blit(fuel_value, (panel_rect.right - 50, 100))
        
        # Fuel bar background
        fuel_bar_bg = pygame.Rect(20, 125, 270, 15)
        pygame.draw.rect(self.screen, (40, 40, 40), fuel_bar_bg)
        
        # Fuel bar fill
        fuel_bar_fill = pygame.Rect(20, 125, int(270 * (self.fuel_level / 100)), 15)
        fuel_fill_color = DANGER_COLOR if self.fuel_level < 20 else HUD_COLOR
        pygame.draw.rect(self.screen, fuel_fill_color, fuel_bar_fill)
        
        # System symbols
        symbols_text = self.font_small.render("SYSTEM SYMBOLS", True, HUD_COLOR)
        self.screen.blit(symbols_text, (20, 160))
        
        y_offset = 190
        for symbol in self.symbols:
            # Determine if this symbol should be highlighted
            highlighted = any(h["type"] == "symbol" and h["data"]["position"] == symbol.position 
                            for h in self.highlights)
            
            # Symbol color based on status and highlight
            symbol_color = WARNING_COLOR if symbol.status == "warning" else HUD_COLOR
            if highlighted:
                symbol_color = HIGHLIGHT_COLOR
                
            symbol_text = self.font_symbols.render(symbol.symbol, True, symbol_color)
            self.screen.blit(symbol_text, (30, y_offset))
            
            # Position and status
            status_color = WARNING_COLOR if symbol.status == "warning" else HUD_COLOR
            status_text = self.font_tiny.render(f"{symbol.position} - {symbol.status}", True, status_color)
            self.screen.blit(status_text, (60, y_offset + 5))
            
            y_offset += 40
        
        # Tactical shapes key
        shapes_text = self.font_small.render("TACTICAL SHAPES", True, HUD_COLOR)
        self.screen.blit(shapes_text, (20, 310))
        
        # Draw small examples of each shape type
        shape_types = [s["type"] for s in SHAPES]
        x_offset = 30
        for shape_type in shape_types:
            # Create a mini shape for the legend
            mini_shape = Shape(0, shape_type, HUD_COLOR, x_offset, 345, 8)
            mini_shape.draw(self.screen)
            x_offset += 50
    
    def draw_center_view(self):
        """Draw the center view with reticle and shapes"""
        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        # Draw targeting reticle
        pygame.draw.circle(self.screen, HUD_COLOR, (center_x, center_y), 40, 1)
        pygame.draw.circle(self.screen, HUD_COLOR, (center_x, center_y), 2)
        
        # Draw crosshairs
        pygame.draw.line(self.screen, HUD_COLOR, (center_x, center_y - 50), (center_x, center_y - 10), 1)
        pygame.draw.line(self.screen, HUD_COLOR, (center_x, center_y + 10), (center_x, center_y + 50), 1)
        pygame.draw.line(self.screen, HUD_COLOR, (center_x - 50, center_y), (center_x - 10, center_y), 1)
        pygame.draw.line(self.screen, HUD_COLOR, (center_x + 10, center_y), (center_x + 50, center_y), 1)
        
        # Draw active shapes
        for shape in self.shapes:
            # Check if shape is highlighted
            highlighted = any(h["type"] == "shape" and h["data"]["id"] == shape.id 
                            for h in self.highlights)
            
            # Use highlight color if highlighted
            color = HIGHLIGHT_COLOR if highlighted else shape.color
            
            # Create a temporary shape with the correct color for drawing
            temp_shape = Shape(shape.id, shape.type, color, shape.x, shape.y, shape.size)
            temp_shape.draw(self.screen)
            
    def draw_right_panel(self):
        """Draw the right panel with change log"""
        # Panel area
        panel_rect = pygame.Rect(SCREEN_WIDTH - 310, 60, 300, SCREEN_HEIGHT - 110)
        pygame.draw.rect(self.screen, HUD_COLOR, panel_rect, 1)
        
        # Changes log header
        log_text = self.font_medium.render("CHANGES LOG", True, HUD_COLOR)
        self.screen.blit(log_text, (panel_rect.left + 10, 70))
        
        # Export CSV button
        export_button = pygame.Rect(panel_rect.right - 110, 70, 100, 25)
        pygame.draw.rect(self.screen, (0, 100, 0), export_button)
        pygame.draw.rect(self.screen, HUD_COLOR, export_button, 1)
        
        export_text = self.font_small.render("Export CSV", True, HUD_COLOR)
        self.screen.blit(export_text, (export_button.left + 10, export_button.top + 5))
        
        # Changes list
        y_offset = 110
        for i, change in enumerate(self.change_tracker.changes[:10]):  # Show only top 10 changes
            # Determine if this change is highlighted (recent)
            highlighted = any(h["data"] == change for h in self.highlights)
            
            # Draw change box
            change_rect = pygame.Rect(panel_rect.left + 10, y_offset, 280, 65)
            
            # Semi-transparent background
            change_bg = pygame.Surface((280, 65), pygame.SRCALPHA)
            bg_color = (0, 100, 50, 100) if highlighted else (0, 0, 0, 100)
            change_bg.fill(bg_color)
            self.screen.blit(change_bg, (panel_rect.left + 10, y_offset))
            
            pygame.draw.rect(self.screen, HUD_COLOR, change_rect, 1)
            
            # Type and time
            header_color = HIGHLIGHT_COLOR if highlighted else HUD_COLOR
            type_text = self.font_small.render(change["type"], True, header_color)
            self.screen.blit(type_text, (change_rect.left + 5, y_offset + 5))
            
            time_display = datetime.fromisoformat(change["time"]).strftime("%H:%M:%S")
            time_text = self.font_tiny.render(time_display, True, (150, 150, 150))
            self.screen.blit(time_text, (change_rect.right - time_text.get_width() - 5, y_offset + 5))
            
            # Content depends on change type
            if change["type"] in ["altitude", "speed", "heading", "fuel_level", "temperature"]:
                # Old value
                old_text = self.font_tiny.render(f"Old: {change['oldValue']}", True, HUD_COLOR)
                self.screen.blit(old_text, (change_rect.left + 5, y_offset + 25))
                
                # New value
                new_text = self.font_tiny.render(f"New: {change['newValue']}", True, WARNING_COLOR)
                self.screen.blit(new_text, (change_rect.right - new_text.get_width() - 5, y_offset + 25))
                
                # Change amount
                if "diff" in change:
                    diff = change["diff"]
                    diff_color = (0, 255, 0) if diff > 0 else (255, 0, 0)
                    diff_text = self.font_tiny.render(
                        f"Change: {'+' if diff > 0 else ''}{diff}", True, diff_color)
                    self.screen.blit(diff_text, (change_rect.left + 5, y_offset + 45))
                    
                    if "diffPercent" in change:
                        pct_text = self.font_tiny.render(
                            f"({'+' if diff > 0 else ''}{change['diffPercent']}%)", True, diff_color)
                        self.screen.blit(pct_text, (change_rect.left + diff_text.get_width() + 10, y_offset + 45))
                        
            elif change["type"] == "symbol":
                pos_text = self.font_tiny.render(f"Position: {change['position']}", True, HUD_COLOR)
                self.screen.blit(pos_text, (change_rect.left + 5, y_offset + 25))
                
                old_text = self.font_tiny.render(f"Old: {change['oldValue']}", True, HUD_COLOR)
                self.screen.blit(old_text, (change_rect.left + 5, y_offset + 45))
                
                new_text = self.font_tiny.render(f"New: {change['newValue']}", True, WARNING_COLOR)
                self.screen.blit(new_text, (change_rect.right - new_text.get_width() - 5, y_offset + 45))
                
            elif change["type"] == "symbolStatus":
                pos_text = self.font_tiny.render(f"Position: {change['position']}", True, HUD_COLOR)
                self.screen.blit(pos_text, (change_rect.left + 5, y_offset + 25))
                
                old_text = self.font_tiny.render(f"Old status: {change['oldValue']}", True, HUD_COLOR)
                self.screen.blit(old_text, (change_rect.left + 5, y_offset + 45))
                
                status_color = WARNING_COLOR if change['newValue'] == 'warning' else HUD_COLOR
                new_text = self.font_tiny.render(f"New status: {change['newValue']}", True, status_color)
                self.screen.blit(new_text, (change_rect.right - new_text.get_width() - 5, y_offset + 45))
                
            elif change["type"] == "shape":
                id_text = self.font_tiny.render(f"Shape ID: {change['id']}", True, HUD_COLOR)
                self.screen.blit(id_text, (change_rect.left + 5, y_offset + 25))
                
                type_change = self.font_tiny.render(
                    f"{change['oldValue']} → {change['newValue']}", True, WARNING_COLOR)
                self.screen.blit(type_change, (change_rect.left + 100, y_offset + 25))
                
                pos_text = self.font_tiny.render(
                    f"Pos: {change['oldPosition']} → {change['newPosition']}", True, WARNING_COLOR)
                self.screen.blit(pos_text, (change_rect.left + 5, y_offset + 45))
                
            y_offset += 75
            
            # Stop if we've reached the bottom of the panel
            if y_offset > panel_rect.bottom - 20:
                break
    
    def draw_bottom_bar(self):
        """Draw the bottom status bar"""
        # Bottom line
        pygame.draw.line(self.screen, HUD_COLOR, (10, SCREEN_HEIGHT - 50), (SCREEN_WIDTH - 10, SCREEN_HEIGHT - 50), 2)
        
        # System active status
        active_text = self.font_small.render("SYSTEM ACTIVE", True, HUD_COLOR)
        self.screen.blit(active_text, (20, SCREEN_HEIGHT - 35))
        
        # CSV record count
        csv_text = self.font_small.render(f"CSV DATA: {len(self.change_tracker.csv_data)} RECORDS", True, HUD_COLOR)
        self.screen.blit(csv_text, (SCREEN_WIDTH // 2 - csv_text.get_width() // 2, SCREEN_HEIGHT - 35))
        
        # Warning status
        has_warning = any(s.status == "warning" for s in self.symbols)
        status_color = WARNING_COLOR if has_warning else HUD_COLOR
        status_text = self.font_small.render(f"STATUS: {'WARNING' if has_warning else 'NORMAL'}", True, status_color)
        self.screen.blit(status_text, (SCREEN_WIDTH - 20 - status_text.get_width(), SCREEN_HEIGHT - 35))
    
    def check_events(self):
        """Check for pygame events (keyboard, mouse, etc)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    # Export CSV on 'C' key press
                    self.export_csv()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if export button was clicked
                export_button = pygame.Rect(SCREEN_WIDTH - 110, 70, 100, 25)
                if export_button.collidepoint(event.pos):
                    self.export_csv()
    
    def export_csv(self):
        """Export data to CSV file"""
        filename = self.change_tracker.export_to_csv()
        print(f"Data exported to {filename}")
        
    def update_highlights(self):
        """Update the highlight effects"""
        current_time = pygame.time.get_ticks()
        
        # Remove highlights after a certain time
        self.highlights = [h for h in self.highlights 
                         if current_time - h["time"] < 3000]  # 3 second highlight
    
    def run(self):
        """Main game loop"""
        try:
            while self.running:
                self.check_events()
                self.update_hud_data()
                self.update_highlights()
                
                # Draw everything
                self.draw_hud()
                
                pygame.display.flip()
                self.clock.tick(FPS)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            pygame.quit()


if __name__ == "__main__":
    # Create and run the HUD
    hud = HUD()
    hud.run()