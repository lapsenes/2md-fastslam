import random
from utils import generate_movement_noise

class Robot:
    def __init__(self, env):
        # Initialize with random valid float position
        while True:
            self.x = random.uniform(1, env.size-2)
            self.y = random.uniform(1, env.size-2)
            # Check grid position for collision
            grid_x = int(round(self.x))
            grid_y = int(round(self.y))
            if env.is_valid_position(grid_x, grid_y):
                break
                
        self.movement_deltas = {
            'Up': (0, -1),
            'Down': (0, 1),
            'Left': (-1, 0),
            'Right': (1, 0)
        }
        self.measurements = {
            'Up': {'distance': 0, 'obstacle': None},
            'Down': {'distance': 0, 'obstacle': None},
            'Left': {'distance': 0, 'obstacle': None},
            'Right': {'distance': 0, 'obstacle': None}
        }
        self.show_measurements = False  # Add flag for measurement visualization

    def set_position(self, x, y):
        self.x = x
        self.y = y

    @property
    def position(self):
        return (self.x, self.y)

    def try_move(self, direction, env):
        if direction not in self.movement_deltas:
            return False
            
        dx, dy = self.movement_deltas[direction]
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        if env.is_valid_position(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.show_measurements = False  # Hide measurements after movement
            print(f"Robot moved to ({self.x:.2f}, {self.y:.2f}) without noise")
            return True
            
        return False

    def measure_environment(self, env):
        """Measure distance to nearest obstacle edges"""
        current_x = self.x
        current_y = self.y
        rounded_x = int(round(current_x))
        rounded_y = int(round(current_y))
        
        # Reset measurements
        self.measurements = {
            'Up': {'distance': 0, 'obstacle': None},
            'Down': {'distance': 0, 'obstacle': None},
            'Left': {'distance': 0, 'obstacle': None},
            'Right': {'distance': 0, 'obstacle': None}
        }
        
        # Up (decreasing y)
        for y in range(rounded_y, -1, -1):  # Start from current rounded position
            if env.grid[y, rounded_x] == 1:
                distance = current_y - y - 0.5  # Distance to edge
                if distance > 0:  # Only record if distance is positive
                    self.measurements['Up'].update({
                        'distance': distance,
                        'obstacle': (rounded_x, y)
                    })
                break
        
        # Down (increasing y)
        for y in range(rounded_y, env.size):  # Start from current rounded position
            if env.grid[y, rounded_x] == 1:
                distance = y - current_y - 0.5
                if distance > 0:
                    self.measurements['Down'].update({
                        'distance': distance,
                        'obstacle': (rounded_x, y)
                    })
                break
        
        # Left (decreasing x)
        for x in range(rounded_x, -1, -1):  # Start from current rounded position
            if env.grid[rounded_y, x] == 1:
                distance = current_x - x - 0.5
                if distance > 0:
                    self.measurements['Left'].update({
                        'distance': distance,
                        'obstacle': (x, rounded_y)
                    })
                break
        
        # Right (increasing x)
        for x in range(rounded_x, env.size):  # Start from current rounded position
            if env.grid[rounded_y, x] == 1:
                distance = x - current_x - 0.5
                if distance > 0:
                    self.measurements['Right'].update({
                        'distance': distance,
                        'obstacle': (x, rounded_y)
                    })
                break
        
        print("\nMeasurements:")
        for direction, data in self.measurements.items():
            if data['obstacle']:
                print(f"{direction}:")
                print(f"  Distance: {data['distance']:.2f}")
                print(f"  Grid position: {data['obstacle']}")
        
        self.show_measurements = True
        return self.measurements