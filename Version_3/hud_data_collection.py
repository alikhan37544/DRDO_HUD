import pygame
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from colorama import Fore, Back, Style, init
from hud_representaion import HUDData, update_hud_values

# Initialize colorama for colored terminal output
init(autoreset=True)

class HUDAnalyzer:
    def __init__(self, history_length=30):
        self.data_history = deque(maxlen=history_length)
        self.current_data = HUDData()
        self.prev_data = None
        self.history_length = history_length
        self.start_time = datetime.now()
        self.change_thresholds = {
            'altitude': 50,       # feet
            'airspeed': 5,        # knots
            'heading': 5,         # degrees
            'pitch': 2,           # degrees
            'roll': 5,            # degrees
            'g_force': 0.3,       # G
            'aoa': 1.0,           # degrees
            'mach': 0.05,         # mach
            'fuel': 2,            # percent
            'target_distance': 1, # km
            'waypoint_distance': 2 # km
        }
        
        # Create CSV file for data logging
        self.csv_filename = f"hud_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()
        
    def init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'altitude', 'airspeed', 'heading', 'pitch', 'roll',
            'g_force', 'aoa', 'mach', 'fuel', 'weapon_status', 'target_locked',
            'target_distance', 'waypoint_bearing', 'waypoint_distance'
        ]
        
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    def update(self):
        """Update with new HUD data"""
        self.prev_data = self.current_data
        self.current_data = update_hud_values(self.current_data)
        self.data_history.append(self.current_data)
        self.log_to_csv()
        
    def log_to_csv(self):
        """Log current data to CSV"""
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                self.current_data.altitude,
                self.current_data.airspeed,
                self.current_data.heading,
                self.current_data.pitch,
                self.current_data.roll,
                self.current_data.g_force,
                self.current_data.aoa,
                self.current_data.mach,
                self.current_data.fuel,
                self.current_data.weapon_status,
                self.current_data.target_locked,
                self.current_data.target_distance if self.current_data.target_locked else 'N/A',
                self.current_data.waypoint_bearing,
                self.current_data.waypoint_distance
            ])
    
    def calculate_rate_of_change(self, attr):
        """Calculate rate of change for a specific attribute"""
        if len(self.data_history) < 2:
            return 0
        
        current_val = getattr(self.current_data, attr)
        prev_val = getattr(self.prev_data, attr)
        
        # Handle circular values like heading
        if attr == 'heading':
            diff = (current_val - prev_val + 180) % 360 - 180
            return diff
        
        return current_val - prev_val
    
    def get_trend(self, attr, samples=5):
        """Get trend direction over the last few samples"""
        if len(self.data_history) < 2:
            return "stable"
        
        samples = min(samples, len(self.data_history))
        recent_data = list(self.data_history)[-samples:]
        
        if attr == 'heading':
            # Special handling for circular values
            diffs = []
            for i in range(1, len(recent_data)):
                prev = getattr(recent_data[i-1], attr)
                curr = getattr(recent_data[i], attr)
                diff = (curr - prev + 180) % 360 - 180
                diffs.append(diff)
            
            avg_change = sum(diffs) / len(diffs) if diffs else 0
        else:
            values = [getattr(data, attr) for data in recent_data]
            diffs = [values[i] - values[i-1] for i in range(1, len(values))]
            avg_change = sum(diffs) / len(diffs) if diffs else 0
        
        threshold = self.change_thresholds.get(attr, 0.01) / 2
        
        if abs(avg_change) < threshold:
            return "stable"
        return "increasing" if avg_change > 0 else "decreasing"
    
    def get_anomaly(self, attr):
        """Check if current value is anomalous compared to recent history"""
        if len(self.data_history) < 10:  # Need enough history
            return False
        
        recent_data = list(self.data_history)[-10:]
        values = [getattr(data, attr) for data in recent_data[:-1]]  # Exclude current
        
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        current = getattr(self.current_data, attr)
        z_score = (current - mean) / std_dev if std_dev else 0
        
        return abs(z_score) > 2  # More than 2 standard deviations
    
    def get_inference(self, attr):
        """Generate inference text based on attribute and its changes"""
        current_val = getattr(self.current_data, attr)
        
        if self.prev_data is None:
            return f"{attr}: {current_val:.2f} (initial reading)"
        
        prev_val = getattr(self.prev_data, attr)
        change = self.calculate_rate_of_change(attr)
        trend = self.get_trend(attr)
        is_anomaly = self.get_anomaly(attr)
        
        inference = ""
        
        # Basic change reporting
        if attr == 'heading':
            # Special handling for circular values
            if abs(change) > self.change_thresholds.get(attr, 0):
                turn_dir = "right" if change > 0 else "left"
                inference = f"Aircraft turning {turn_dir} at {abs(change):.1f}° per second"
                
                if abs(change) > 3:
                    inference += " (sharp turn)"
        
        elif attr == 'altitude':
            if abs(change) > self.change_thresholds.get(attr, 0):
                direction = "climbing" if change > 0 else "descending"
                inference = f"Aircraft {direction} at {abs(change):.1f} feet per second"
                
                if abs(change) > 100:
                    inference += " (rapid altitude change)"
        
        elif attr == 'airspeed':
            if abs(change) > self.change_thresholds.get(attr, 0):
                direction = "accelerating" if change > 0 else "decelerating"
                inference = f"Aircraft {direction} at {abs(change):.1f} knots per second"
                
                if self.current_data.airspeed > 550:
                    inference += " (high speed flight)"
                elif self.current_data.airspeed < 200:
                    inference += " (low speed flight - caution)"
        
        elif attr == 'pitch':
            if abs(change) > self.change_thresholds.get(attr, 0):
                direction = "nose up" if change > 0 else "nose down"
                inference = f"Aircraft pitching {direction} at {abs(change):.1f}° per second"
                
                if abs(current_val) > 20:
                    inference += f" (extreme pitch angle: {current_val:.1f}°)"
        
        elif attr == 'roll':
            if abs(change) > self.change_thresholds.get(attr, 0):
                direction = "right" if change > 0 else "left"
                inference = f"Aircraft rolling to {direction} at {abs(change):.1f}° per second"
                
                if abs(current_val) > 30:
                    inference += f" (steep bank angle: {current_val:.1f}°)"
        
        elif attr == 'g_force':
            if abs(change) > self.change_thresholds.get(attr, 0):
                inference = f"G-force changing at {abs(change):.2f}G per second"
                
                if current_val > 5:
                    inference += f" (high G maneuver: {current_val:.1f}G)"
                elif current_val < 0:
                    inference += f" (negative G situation: {current_val:.1f}G)"
        
        elif attr == 'aoa':
            if abs(change) > self.change_thresholds.get(attr, 0):
                inference = f"Angle of attack changing at {abs(change):.1f}° per second"
                
                if current_val > 15:
                    inference += f" (approaching stall angle: {current_val:.1f}°)"
        
        elif attr == 'mach':
            if abs(change) > self.change_thresholds.get(attr, 0):
                direction = "increasing" if change > 0 else "decreasing"
                inference = f"Mach number {direction} at {abs(change):.2f} per second"
                
                if current_val > 1.0:
                    inference += " (supersonic flight)"
        
        elif attr == 'fuel':
            burn_rate = abs(change) * 3600  # Convert to per hour
            inference = f"Fuel consumption rate: {burn_rate:.1f}% per hour"
            
            if current_val < 20:
                inference += f" (LOW FUEL WARNING: {current_val:.1f}%)"
            
            remaining_minutes = (current_val / burn_rate) * 60 if burn_rate > 0 else float('inf')
            if remaining_minutes < float('inf'):
                inference += f" (Est. {remaining_minutes:.0f} minutes remaining)"
        
        elif attr == 'target_distance':
            if self.current_data.target_locked:
                if prev_val is not None:
                    closing_rate = prev_val - current_val
                    inference = f"Target distance: {current_val:.1f} km"
                    if abs(closing_rate) > 0:
                        status = "closing" if closing_rate > 0 else "moving away"
                        inference += f", {status} at {abs(closing_rate):.1f} km/s"
                        
                        time_to_intercept = current_val / closing_rate if closing_rate > 0 else float('inf')
                        if time_to_intercept < 60 and time_to_intercept > 0:
                            inference += f" (intercept in {time_to_intercept:.0f} seconds)"
            else:
                inference = "No target locked"
        
        elif attr == 'waypoint_distance':
            if prev_val is not None:
                closing_rate = prev_val - current_val
                inference = f"Distance to waypoint: {current_val:.1f} km"
                
                time_to_wp = current_val / closing_rate if closing_rate > 0 else float('inf')
                if time_to_wp < 300 and time_to_wp > 0:
                    inference += f" (ETA: {time_to_wp/60:.1f} minutes)"
                    
                    if time_to_wp < 60:
                        inference += " - waypoint approaching"
        
        # Highlight anomalies or concerning values
        if is_anomaly and not inference:
            inference = f"Unusual {attr} value detected: {current_val:.2f}"
        
        return inference if inference else f"{attr}: {current_val:.2f} ({trend})"
    
    def format_output(self, text, attr, highlight=False):
        """Format text with appropriate colors based on attribute and change significance"""
        prefix = ""
        
        # Determine if change is significant
        if self.prev_data is not None:
            change = abs(self.calculate_rate_of_change(attr))
            threshold = self.change_thresholds.get(attr, 0.01)
            
            is_significant = change > threshold
            is_anomaly = self.get_anomaly(attr)
            
            # Set prefix based on significance
            if is_anomaly:
                prefix = f"{Fore.RED}{Back.YELLOW}[ANOMALY]{Style.RESET_ALL} "
            elif is_significant:
                if attr in ['fuel'] and getattr(self.current_data, attr) < 20:
                    prefix = f"{Fore.RED}[WARNING]{Style.RESET_ALL} "
                else:
                    prefix = f"{Fore.YELLOW}[CHANGE]{Style.RESET_ALL} "
        
        # Apply highlighting
        if highlight:
            return prefix + f"{Fore.CYAN}{text}{Style.RESET_ALL}"
        elif attr in ['altitude', 'airspeed'] and self.prev_data is not None:
            current = getattr(self.current_data, attr)
            prev = getattr(self.prev_data, attr)
            
            if current > prev:
                return prefix + f"{Fore.GREEN}{text}{Style.RESET_ALL}"
            elif current < prev:
                return prefix + f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"
        
        return prefix + text
    
    def print_status(self):
        """Print comprehensive status report with inferences"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print header
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"{Fore.WHITE}{Back.BLUE}============= HUD DATA ANALYSIS =============")
        print(f"Time Elapsed: {elapsed:.1f} seconds{Style.RESET_ALL}\n")
        
        # Flight parameters section
        print(f"{Fore.WHITE}{Back.GREEN}===== FLIGHT PARAMETERS ====={Style.RESET_ALL}")
        
        # Basic flight data
        altitude_inf = self.get_inference('altitude')
        airspeed_inf = self.get_inference('airspeed')
        heading_inf = self.get_inference('heading')
        
        print(self.format_output(altitude_inf, 'altitude', highlight=True))
        print(self.format_output(airspeed_inf, 'airspeed', highlight=True))
        print(self.format_output(heading_inf, 'heading', highlight=True))
        
        # Attitude data
        print(f"\n{Fore.WHITE}{Back.CYAN}===== AIRCRAFT ATTITUDE ====={Style.RESET_ALL}")
        
        pitch_inf = self.get_inference('pitch')
        roll_inf = self.get_inference('roll')
        g_force_inf = self.get_inference('g_force')
        aoa_inf = self.get_inference('aoa')
        
        print(self.format_output(pitch_inf, 'pitch'))
        print(self.format_output(roll_inf, 'roll'))
        print(self.format_output(g_force_inf, 'g_force'))
        print(self.format_output(aoa_inf, 'aoa'))
        
        # Performance data
        print(f"\n{Fore.WHITE}{Back.MAGENTA}===== PERFORMANCE DATA ====={Style.RESET_ALL}")
        
        mach_inf = self.get_inference('mach')
        fuel_inf = self.get_inference('fuel')
        
        print(self.format_output(mach_inf, 'mach'))
        print(self.format_output(fuel_inf, 'fuel'))
        
        # Navigation & targeting
        print(f"\n{Fore.WHITE}{Back.YELLOW}===== NAVIGATION & TARGETING ====={Style.RESET_ALL}")
        
        # Weapon status
        print(f"Weapon Status: {self.current_data.weapon_status}")
        
        # Target info
        if self.current_data.target_locked:
            target_inf = self.get_inference('target_distance')
            print(f"{Fore.RED}[TARGET LOCKED] {target_inf}{Style.RESET_ALL}")
        else:
            print("No target currently locked")
        
        # Waypoint info
        wp_inf = self.get_inference('waypoint_distance')
        print(self.format_output(wp_inf, 'waypoint_distance'))
        print(f"Waypoint Bearing: {self.current_data.waypoint_bearing:.1f}°")
        
        # Flight situation inference
        print(f"\n{Fore.WHITE}{Back.RED}===== FLIGHT SITUATION ANALYSIS ====={Style.RESET_ALL}")
        
        # Analyze overall situation
        self.analyze_flight_situation()
        
        # Data storage notification
        print(f"\n{Fore.WHITE}Data logged to: {self.csv_filename}{Style.RESET_ALL}")

    def analyze_flight_situation(self):
        """Analyze overall flight situation and provide comprehensive inference"""
        # Skip if not enough data
        if len(self.data_history) < 5:
            print("Collecting flight data... Standby for analysis")
            return
        
        # Extract current values for easier access
        alt = self.current_data.altitude
        spd = self.current_data.airspeed
        pitch = self.current_data.pitch
        roll = self.current_data.roll
        g = self.current_data.g_force
        aoa = self.current_data.aoa
        fuel = self.current_data.fuel
        
        # Get trends
        alt_trend = self.get_trend('altitude')
        spd_trend = self.get_trend('airspeed')
        roll_trend = self.get_trend('roll')
        pitch_trend = self.get_trend('pitch')
        
        # Determine flight condition
        flight_conditions = []
        
        # Altitude-based conditions
        if alt < 1000:
            flight_conditions.append(f"{Fore.RED}LOW ALTITUDE FLIGHT ({alt:.0f} ft){Style.RESET_ALL}")
        elif alt > 30000:
            flight_conditions.append(f"High altitude flight ({alt:.0f} ft)")
        
        # Speed-based conditions
        if spd < 180:
            flight_conditions.append(f"{Fore.RED}LOW AIRSPEED ({spd:.0f} knots) - STALL RISK{Style.RESET_ALL}")
        elif spd > 600:
            flight_conditions.append(f"High speed flight ({spd:.0f} knots)")
        
        # Maneuver detection
        if abs(roll) > 30 and abs(g) > 2:
            flight_conditions.append(f"{Fore.YELLOW}Banking turn maneuver ({g:.1f}G, {roll:.1f}° roll){Style.RESET_ALL}")
        
        if abs(pitch) > 15 and alt_trend == "increasing":
            flight_conditions.append(f"{Fore.YELLOW}Climbing maneuver ({pitch:.1f}° pitch up){Style.RESET_ALL}")
        
        if abs(pitch) > 15 and alt_trend == "decreasing":
            flight_conditions.append(f"{Fore.YELLOW}Descending maneuver ({pitch:.1f}° pitch down){Style.RESET_ALL}")
        
        # Fixed line - removed abs() from roll_trend which is a string
        if roll_trend == "increasing" and abs(roll) > 20:
            flight_conditions.append(f"{Fore.YELLOW}Roll initiation{Style.RESET_ALL}")
        
        if aoa > 12:
            flight_conditions.append(f"{Fore.RED}HIGH ANGLE OF ATTACK ({aoa:.1f}°) - STALL RISK{Style.RESET_ALL}")
        
        # Combat situation assessment
        if self.current_data.target_locked:
            dist = self.current_data.target_distance
            if dist < 5:
                flight_conditions.append(f"{Fore.RED}CLOSE COMBAT ENGAGEMENT (Target at {dist:.1f}km){Style.RESET_ALL}")
            else:
                flight_conditions.append(f"Target engagement (Target at {dist:.1f}km)")
        
        # Fuel situation
        if fuel < 15:
            flight_conditions.append(f"{Fore.RED}CRITICAL FUEL STATE ({fuel:.1f}%){Style.RESET_ALL}")
        elif fuel < 25:
            flight_conditions.append(f"{Fore.YELLOW}Low fuel warning ({fuel:.1f}%){Style.RESET_ALL}")
        
        # Print flight conditions
        if flight_conditions:
            for condition in flight_conditions:
                print(f"• {condition}")
        else:
            print("• Normal flight conditions")
        
        # Overall mission status
        wp_dist = self.current_data.waypoint_distance
        if wp_dist < 5:
            print(f"\n{Fore.GREEN}Approaching waypoint ({wp_dist:.1f}km remaining){Style.RESET_ALL}")
        elif wp_dist < 0.5:
            print(f"\n{Fore.GREEN}Waypoint reached!{Style.RESET_ALL}")


def main():
    analyzer = HUDAnalyzer()
    
    try:
        while True:
            analyzer.update()
            analyzer.print_status()
            time.sleep(0.5)  # Update every half second
    except KeyboardInterrupt:
        print(f"\n{Fore.GREEN}Data collection stopped. Data saved to {analyzer.csv_filename}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()