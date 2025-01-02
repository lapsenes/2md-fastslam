import numpy as np

def generate_movement_noise(movement_distance=1.0, uncertainty=0.2):
    # Standard deviation based on movement distance
    std_dev = uncertainty * movement_distance  # 0.2 variance for 1.0 movement
    
    # Generate noise
    noise_x = np.random.normal(0, std_dev)
    noise_y = np.random.normal(0, std_dev)
    
    return noise_x, noise_y
