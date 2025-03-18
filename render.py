import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# Load the simulation data from the CSV file
def load_simulation_data(filename="simulation_data.csv"):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row
        for row in reader:
            time = float(row[0])
            positions = np.array(row[1:], dtype=float).reshape(-1, 2)
            data.append((time, positions))
    return data

# Create the animation
def render_animation(data):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    # Initialize the particle patches
    num_particles = len(data[0][1])
    particles = [plt.Circle((0, 0), 0.02, fc='blue', ec='black') for _ in range(num_particles)]
    for particle in particles:
        ax.add_patch(particle)

    def update(frame):
        time, positions = data[frame]  # Get the data for the current frame
        for i, particle in enumerate(particles):
            particle.set_center(positions[i])  # Update particle positions
        return particles

    # Use FuncAnimation to create the animation
    ani = FuncAnimation(fig, update, frames=len(data), interval=50, blit=True)
    plt.show()

# Load the data and render the animation
data = load_simulation_data("simulation_data.csv")
render_animation(data)
