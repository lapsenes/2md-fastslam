import random
import numpy as np
from utils import generate_movement_noise
import math

class Particle:
    def __init__(self, env, robot_x=None, robot_y=None):
        # Initialize position near robot if provided, otherwise random
        if robot_x is not None and robot_y is not None:
            # Generate position within ±5.0 of robot's position
            while True:
                offset_x = random.uniform(-5.0, 5.0)
                offset_y = random.uniform(-5.0, 5.0)
                self.x = robot_x + offset_x
                self.y = robot_y + offset_y
                # Ensure position is within grid bounds and valid
                if (0 < self.x < env.size-1 and 
                    0 < self.y < env.size-1 and 
                    env.is_valid_position(int(round(self.x)), int(round(self.y)))):
                    break
        else:
            # Original random initialization
            while True:
                self.x = random.uniform(1, env.size-2)
                self.y = random.uniform(1, env.size-2)
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
        
        old_x, old_y = self.x, self.y
        self.x += dx + noise_x
        self.y += dy + noise_y
        
        # Keep particles within grid bounds
        self.x = max(0, min(self.x, env.size - 1))
        self.y = max(0, min(self.y, env.size - 1))
        
        # Debug print statement
        print(f"Particle moved from ({old_x:.2f}, {old_y:.2f}) to ({self.x:.2f}, {self.y:.2f}) with noise ({noise_x:.2f}, {noise_y:.2f})")

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

    def calculate_measurement_difference(self, landmark_id, new_expected_pos):
        """Calculate difference between new and old expected positions"""
        old_pos = self.landmarks[landmark_id]['expected_position']
        dx = new_expected_pos[0] - old_pos[0]
        dy = new_expected_pos[1] - old_pos[1]
        Y = (dx, dy)
        return Y

    def calculate_Q(self, measurement_distance, uncertainty=0.2):
        """Calculate measurement uncertainty matrix Q"""
        Q_x = measurement_distance * uncertainty
        Q_y = measurement_distance * uncertainty
        return (Q_x, Q_y)

    def calculate_S(self, prev_Q, current_Q):
        """Calculate combined uncertainty P from previous and current Q"""
        S_x = prev_Q[0] + current_Q[0]
        S_y = prev_Q[1] + current_Q[1]
        return (S_x, S_y)

    def calculate_kalman_gain(self, S, prev_Q, current_Q):
        """Calculate Kalman gain K as ratio of previous to current uncertainty"""
        K_x = prev_Q[0] * 1/(S[0]) 
        K_y = prev_Q[1] * 1/(S[1]) 
        return (K_x, K_y)

    def calculate_updated_position(self, old_pos, Y, K):
        """Calculate updated position using Kalman filter equation"""
        new_x = old_pos[0] + Y[0] * K[0]
        new_y = old_pos[1] + Y[1] * K[1]
        return (new_x, new_y)

    def calculate_weight(self, Y, S):
        """Calculate particle weight using a Gaussian function"""
        Y = np.array(Y)
        S = np.diag(S)  # Convert S to a 2x2 diagonal matrix
        
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        
        # Calculate exponent term
        exp_term = -0.5 * np.dot(np.dot(Y.T, inv_S), Y)
        # Calculate weight
        weight = (1 / (2 * math.pi * math.sqrt(det_S))) * math.exp(exp_term)
        
        return weight

    def register_single_measurement(self, direction, measurement_data):
        """Process a single direction measurement"""
        if measurement_data['distance'] > 0 and measurement_data['obstacle']:
            landmark_id = measurement_data['obstacle']
            expected_pos = self.calculate_expected_position(direction, measurement_data['distance'])
            current_Q = self.calculate_Q(measurement_data['distance'])
            
            if landmark_id in self.landmarks:
                # Calculate all Kalman filter components
                Y = self.calculate_measurement_difference(landmark_id, expected_pos)
                prev_Q = self.landmarks[landmark_id]['Q']
                S = self.calculate_S(prev_Q, current_Q)
                K = self.calculate_kalman_gain(S, prev_Q, current_Q)
                
                # Calculate updated position and weight
                old_pos = self.landmarks[landmark_id]['expected_position']
                updated_pos = self.calculate_updated_position(old_pos, Y, K)
                self.weight = self.calculate_weight(Y, S)
                
                # Update Q
                updated_Q_x = (1 - K[0]) * prev_Q[0]
                updated_Q_y = (1 - K[1]) * prev_Q[1]
                updated_Q = (updated_Q_x, updated_Q_y)
                
                # Update landmark data
                self.landmarks[landmark_id].update({
                    'expected_position': updated_pos,
                    'Q': updated_Q
                })
                
                # Debug print statements
                print(f"Particle at ({self.x:.2f}, {self.y:.2f}) updated landmark {landmark_id}:")
                print(f"  Direction: {direction}")
                print(f"  Measurement difference Y: {Y}")
                print(f"  Calculated weight: {self.weight:.6f}")
                print(f"  Expected position: {expected_pos}")
                print(f"  Distance to obstacle: {measurement_data['distance']}")
            else:
                # Store new landmark
                self.landmarks[landmark_id] = {
                    'distance': measurement_data['distance'],
                    'direction': direction,
                    'expected_position': expected_pos,
                    'Q': current_Q
                }
                self.weight = 1.0  # New landmarks don't affect weight

                # Debug print statements for new landmark
                print(f"Particle at ({self.x:.2f}, {self.y:.2f}) registered new landmark {landmark_id}:")
                print(f"  Direction: {direction}")
                print(f"  Measurement difference Y: (0.0, 0.0)")
                print(f"  Calculated weight: {self.weight:.6f}")
                print(f"  Expected position: {expected_pos}")
                print(f"  Distance to obstacle: {measurement_data['distance']}")

    def copy_from(self, other_particle):
        """Copy all attributes from another particle"""
        self.x = other_particle.x
        self.y = other_particle.y
        self.landmarks = other_particle.landmarks.copy()
        self.weight = 1.0  # Reset weight after resampling

    @staticmethod
    def resample_particles(particles):
        """Resample particles based on weights using roulette wheel selection"""
        if not particles:
            return particles
            
        N = len(particles)
        # Calculate total weight and step size
        total_weight = sum(p.weight for p in particles)
        if total_weight == 0 or total_weight == N:
            return particles
            
        step = total_weight / N
        
        # Generate lottery numbers
        start = random.uniform(0, step)
        lottery_numbers = [start + i * step for i in range(N)]
        
        # Create cumulative weight array
        cumulative_weights = []
        current_sum = 0
        for p in particles:
            current_sum += p.weight
            cumulative_weights.append(current_sum)
        
        # Store original particles
        original_particles = particles.copy()
        
        # Overwrite existing particles with selected ones
        for i, lottery_number in enumerate(lottery_numbers):
            # Find the particle whose cumulative weight range contains the lottery number
            for j, cum_weight in enumerate(cumulative_weights):
                if lottery_number <= cum_weight:
                    # Copy data from selected particle to existing particle
                    particles[i].copy_from(original_particles[j])
                    break
                    
        print(f"Resampled {N} particles")
        return particles


