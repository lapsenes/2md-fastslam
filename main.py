from environment import Environment
from robot import Robot
from particle import Particle
import matplotlib.pyplot as plt

def main():
    # Initialize environment
    grid_size = int(input("Enter grid size (N for an NxN grid): "))
    object_count = int(input("Enter number of objects: "))
    env = Environment(grid_size, object_count)
    
    # Initialize robot with random position
    robot = Robot(env)
    
    # Initialize particles
    particles = [Particle(env) for _ in range(10)]
    
    print("\nUse arrow keys to move the robot.")
    print("Close the window to exit.")
    
    # Show environment with robot and particles
    env.visualize(robot, particles)
    
    # Keep the window open and responsive
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        plt.close('all')

if __name__ == "__main__":
    main()