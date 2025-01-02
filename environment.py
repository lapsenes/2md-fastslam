import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler

class Environment:
    def __init__(self, size, num_objects):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self._add_frame()
        self._add_random_objects(num_objects)
        # Create figure only once during initialization
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def _add_frame(self):
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def _add_random_objects(self, num_objects):
        inner_positions = random.sample(range(self.size * self.size), num_objects)
        for pos in inner_positions:
            row, col = divmod(pos, self.size)
            self.grid[row, col] = 1

    def is_valid_position(self, x, y):
        # Convert float coordinates to grid indices
        grid_x = int(round(x))
        grid_y = int(round(y))
        
        return (0 <= grid_x < self.size and 
                0 <= grid_y < self.size and 
                self.grid[grid_y, grid_x] == 0)  # Note: grid is accessed [y,x]

    def visualize(self, robot=None, particles=None):
        def on_key(event):
            if not robot:
                return
                
            # Handle both arrow keys and WASD
            key_mapping = {
                'up': 'Up',
                'down': 'Down',
                'left': 'Left',
                'right': 'Right',
                'w': 'Up',
                's': 'Down',
                'a': 'Left',
                'd': 'Right'
            }
            
            if event.key.lower() == 'm':  # 'M' key for measurement
                measurements = robot.measure_environment(self)
                if particles:
                    for particle in particles:
                        particle.register_measurement(measurements)
                self.update_visualization(robot, particles)
            elif event.key.lower() in key_mapping:
                direction = key_mapping[event.key.lower()]
                if robot.try_move(direction, self):
                    # Move particles in the same direction
                    if particles:
                        for particle in particles:
                            particle.move(direction, self)  # Pass self (environment)
                    self.update_visualization(robot, particles)
                    
        # Clear existing callbacks and add new one
        self.fig.canvas.mpl_disconnect('key_press_event')
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        self.update_visualization(robot, particles)
        
    def update_visualization(self, robot=None, particles=None):
        self.ax.clear()
        self.ax.imshow(self.grid, cmap='binary', origin='upper', 
                     extent=(-0.5, self.size-0.5, self.size-0.5, -0.5))
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_aspect('equal')
        
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.set_xticks(np.arange(self.size + 1) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(self.size + 1) - 0.5, minor=True)
        
        self.ax.grid(False)
        self.ax.grid(True, which='minor', color='black', linewidth=0.5)
        self.ax.tick_params(which='both', length=0)
        
        if particles is not None:
            particle_x = [p.x for p in particles]
            particle_y = [p.y for p in particles]
            self.ax.scatter(particle_x, particle_y, c='purple', alpha=0.5, s=20)
        
        if robot is not None:
            self.ax.plot(robot.x, robot.y, 'o', color='lightgreen', alpha=0.6, markersize=15)
            
            # Only draw measurements if flag is True
            if hasattr(robot, 'measurements') and robot.show_measurements:
                current_x = robot.x
                current_y = robot.y
                
                # Draw measurement lines
                if robot.measurements['Up']['distance'] > 0:
                    self.ax.plot([current_x, current_x], 
                               [current_y, current_y - robot.measurements['Up']['distance']], 
                               'r--', alpha=0.5)
                    if robot.measurements['Up']['obstacle']:
                        self.ax.plot(*robot.measurements['Up']['obstacle'], 'rx')
                        
                if robot.measurements['Down']['distance'] > 0:
                    self.ax.plot([current_x, current_x], 
                               [current_y, current_y + robot.measurements['Down']['distance']], 
                               'r--', alpha=0.5)
                    if robot.measurements['Down']['obstacle']:
                        self.ax.plot(*robot.measurements['Down']['obstacle'], 'rx')
                        
                if robot.measurements['Left']['distance'] > 0:
                    self.ax.plot([current_x - robot.measurements['Left']['distance'], current_x], 
                               [current_y, current_y], 
                               'r--', alpha=0.5)
                    if robot.measurements['Left']['obstacle']:
                        self.ax.plot(*robot.measurements['Left']['obstacle'], 'rx')
                        
                if robot.measurements['Right']['distance'] > 0:
                    self.ax.plot([current_x + robot.measurements['Right']['distance'], current_x], 
                               [current_y, current_y], 
                               'r--', alpha=0.5)
                    if robot.measurements['Right']['obstacle']:
                        self.ax.plot(*robot.measurements['Right']['obstacle'], 'rx')

        self.ax.set_xlim(-0.5, self.size-0.5)
        self.ax.set_ylim(-0.5, self.size-0.5)
        plt.gca().invert_yaxis()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Add small pause to allow GUI to update

