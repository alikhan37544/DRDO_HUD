import cv2
import numpy as np

def draw_hud(frame, roll=0, pitch=0, altitude=10000, airspeed=300, heading=90):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    # Artificial Horizon
    horizon_length = 200
    roll_rad = np.deg2rad(roll)
    pitch_offset = int(pitch * (height / 90))  # Assuming 90 degrees pitch maps to full height

    horizon_x1 = int(center_x - horizon_length * np.cos(roll_rad))
    horizon_y1 = int(center_y - horizon_length * np.sin(roll_rad)) + pitch_offset
    horizon_x2 = int(center_x + horizon_length * np.cos(roll_rad))
    horizon_y2 = int(center_y + horizon_length * np.sin(roll_rad)) + pitch_offset

    cv2.line(frame, (horizon_x1, horizon_y1), (horizon_x2, horizon_y2), (0, 255, 0), 2)

    # Altitude Indicator
    altitude_text = f'ALT: {altitude} ft'
    cv2.putText(frame, altitude_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Airspeed Indicator
    airspeed_text = f'SPD: {airspeed} knots'
    cv2.putText(frame, airspeed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Heading Indicator (Compass)
    compass_radius = 50
    compass_center = (width - 100, 100)
    cv2.circle(frame, compass_center, compass_radius, (255, 255, 255), 2)
    heading_rad = np.deg2rad(heading)
    heading_x = int(compass_center[0] + compass_radius * np.sin(heading_rad))
    heading_y = int(compass_center[1] - compass_radius * np.cos(heading_rad))
    cv2.line(frame, compass_center, (heading_x, heading_y), (0, 0, 255), 2)
    heading_text = f'HDG: {heading}°'
    cv2.putText(frame, heading_text, (compass_center[0] - 70, compass_center[1] + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated dynamic data (replace with real data as needed)
        roll = 0  # degrees
        pitch = 0  # degrees
        altitude = 10000  # feet
        airspeed = 300  # knots
        heading = 90  # degrees

        frame = draw_hud(frame, roll, pitch, altitude, airspeed, heading)
        out.write(frame)

        cv2.imshow('HUD Simulation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 'input_video.mp4'  # Replace with your input video path
    output_video_path = 'output_video.mp4'  # Replace with your desired output path
    process_video(input_video_path, output_video_path)
