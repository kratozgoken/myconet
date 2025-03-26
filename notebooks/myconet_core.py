import numpy as np

# Define grid size
GRID_SIZE = 10

# Initialize grid with all costs set to 1 (default path cost)
grid = np.ones((GRID_SIZE, GRID_SIZE))

# Random start and goal positions
start = (0, 0)
goal = (GRID_SIZE - 1, GRID_SIZE - 1)

# You can make some cells more 'costly' (simulate traffic congestion)
grid[3][3] = 5
grid[4][4] = 7
grid[2][5] = 10

import matplotlib.pyplot as plt

def visualize_grid(grid, start, goal):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='viridis', origin='upper')
    plt.colorbar(label='Cost')

    # Mark start and goal
    plt.scatter(start[1], start[0], c='green', s=200, label='Start', marker='o')
    plt.scatter(goal[1], goal[0], c='red', s=200, label='Goal', marker='X')
    plt.legend(loc='upper left')
    plt.title("MycoNet Grid Environment")
    plt.grid(False)
    plt.show()

# Call the function
visualize_grid(grid, start, goal)

import random

# Number of agents and max steps
NUM_AGENTS = 50
MAX_STEPS = 100

# Trail intensity map (pheromones)
trail_map = np.zeros((GRID_SIZE, GRID_SIZE))

# Agent movement directions: 4-way (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

import math

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def move_agent(position, grid, trail_map):
    x, y = position
    candidates = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (
            0 <= nx < GRID_SIZE and 
            0 <= ny < GRID_SIZE and 
            not np.isinf(grid[nx][ny])
           ):

            # Distance to goal (lower = better)
            dist_score = -euclidean((nx, ny), goal)
            trail_score = trail_map[nx][ny]
            cost_penalty = -grid[nx][ny]  # Lower cost is better

            total_score = dist_score + trail_score + cost_penalty
            candidates.append(((nx, ny), total_score))

    if not candidates:
        return position

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# Run agents
for _ in range(NUM_AGENTS):
    pos = start
    for _ in range(MAX_STEPS):
        if pos == goal:
            break
        pos = move_agent(pos, grid, trail_map)
        trail_map[pos[0]][pos[1]] += 1  # Increase trail intensity

def visualize_trail(trail_map, start, goal):
    plt.figure(figsize=(6, 6))
    plt.imshow(trail_map, cmap='plasma', origin='upper')
    plt.colorbar(label='Trail Intensity')
    
    # Start and goal markers
    plt.scatter(start[1], start[0], c='green', s=200, label='Start', marker='o')
    plt.scatter(goal[1], goal[0], c='red', s=200, label='Goal', marker='X')
    plt.legend(loc='upper left')
    plt.title("MycoNet Agent Trail Map")
    plt.grid(False)
    plt.show()

# Show it!
visualize_trail(trail_map, start, goal)

