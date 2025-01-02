import random
import numpy as np
from utils import generate_movement_noise

class Particle:
    def __init__(self, env):
        # Initialize with random valid position
        while True:
            self.x = random.uniform(1, env.size-2)
            self.y = random.uniform(1, env.size-2)
            # Check if position is valid using environment's method
            if env.is_valid_position(int(round(self.x)), int(round(self.y))):
                break
                
        self.weight = 1.0
        self.movement_deltas = {
            'Up': (0, -1),
            'Down': (0, 1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }
        self.landmarks = {}  # Dictionary to store observed obstacles: {robot_obstacle_pos: measurement_data}
        self.landmark_counter = 0  # Unique identifier for landmarks

    def move(self, direction, env):  # env parameter kept for consistency
        if direction not in self.movement_deltas:
            return
            
        dx, dy = self.movement_deltas[direction]
        noise_x, noise_y = generate_movement_noise()  
        
        self.x += dx + noise_x
        self.y += dy + noise_y
        
        # Keep particles within grid bounds
        self.x = max(0, min(self.x, env.size - 1))
        self.y = max(0, min(self.y, env.size - 1))

    def calculate_expected_position(self, direction, distance):
        """Calculate expected obstacle position based on particle's position and measurement"""
        if direction == 'Up':
            return (self.x, self.y - distance)
        elif direction == 'Down':
            return (self.x, self.y + distance)
        elif direction == 'Left':
            return (self.x - distance, self.y)
        elif direction == 'Right':
            return (self.x + distance, self.y)

    def register_measurement(self, robot_measurements):
        """Store robot's measurements converted to expected positions from particle's perspective"""
        for direction, data in robot_measurements.items():
            if data['distance'] > 0 and data['obstacle']:  # Check both distance and obstacle existence
                # Use robot's observed obstacle position as landmark ID
                landmark_id = data['obstacle']
                expected_pos = self.calculate_expected_position(direction, data['distance'])
                
                # Only store if it's a new landmark
                if landmark_id not in self.landmarks:
                    self.landmarks[landmark_id] = {
                        'distance': data['distance'],
                        'direction': direction,
                        'expected_position': expected_pos,
                    }
                    print(f"Particle at ({self.x:.2f}, {self.y:.2f}) registered landmark at {landmark_id}:")
                    print(f"  Direction: {direction}")
                    print(f"  Distance: {data['distance']:.2f}")
                    print(f"  Expected position: {expected_pos}")


