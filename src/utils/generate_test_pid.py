"""
Generate a test P&ID diagram using OpenCV.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_blank_image(width=800, height=600):
    """Create a blank white image."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    return image

def draw_tank(image, x, y, width=100, height=200, label="T-101"):
    """Draw a storage tank symbol."""
    # Draw tank body
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 0), 2)
    
    # Add label
    cv2.putText(image, label, (x + 10, y + height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def draw_pump(image, x, y, radius=40, label="P-101"):
    """Draw a pump symbol."""
    # Draw pump circle
    cv2.circle(image, (x, y), radius, (0, 0, 0), 2)
    
    # Draw cross lines inside
    cv2.line(image, (x - radius//2, y), (x + radius//2, y), (0, 0, 0), 2)
    cv2.line(image, (x, y - radius//2), (x, y + radius//2), (0, 0, 0), 2)
    
    # Add label
    cv2.putText(image, label, (x - radius//2, y + radius + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def draw_valve(image, x, y, size=40, label="V-101"):
    """Draw a valve symbol."""
    # Draw valve body
    cv2.rectangle(image, (x, y), (x + size, y + size), (0, 0, 0), 2)
    
    # Draw diagonal line
    cv2.line(image, (x, y), (x + size, y + size), (0, 0, 0), 2)
    
    # Add label
    cv2.putText(image, label, (x - 10, y + size + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def draw_pipe(image, start_point, end_point):
    """Draw a pipe connection."""
    cv2.line(image, start_point, end_point, (0, 0, 0), 2)

def generate_test_pid():
    """Generate a complete test P&ID diagram."""
    # Create blank image
    image = create_blank_image()
    
    # Draw tank
    tank_x, tank_y = 100, 100
    draw_tank(image, tank_x, tank_y)
    
    # Draw pump
    pump_x, pump_y = 300, 400
    draw_pump(image, pump_x, pump_y)
    
    # Draw valve
    valve_x, valve_y = 500, 380
    draw_valve(image, valve_x, valve_y)
    
    # Draw connecting pipes
    # Tank to pump
    draw_pipe(image, 
             (tank_x + 50, tank_y + 200),  # Tank bottom center
             (tank_x + 50, pump_y))        # Vertical pipe
    draw_pipe(image,
             (tank_x + 50, pump_y),        # Horizontal pipe start
             (pump_x - 40, pump_y))        # To pump inlet
    
    # Pump to valve
    draw_pipe(image,
             (pump_x + 40, pump_y),        # From pump outlet
             (valve_x, pump_y))            # To valve inlet
    
    # Save the image
    output_dir = Path("src") / "data" / "test_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "test_pid.png"
    cv2.imwrite(str(output_path), image)
    
    print(f"Test P&ID image generated: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    generated_file = generate_test_pid()
    
    # Display the image
    image = cv2.imread(generated_file)
    cv2.imshow("Test P&ID", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()