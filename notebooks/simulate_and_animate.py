import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Setup ===
GRID_SIZE = 10
grid = np.ones((GRID_SIZE, GRID_SIZE))
trail_map = np.zeros((GRID_SIZE, GRID_SIZE))
start = (0, 0)
goal = (GRID_SIZE - 1, GRID_SIZE - 1)
DECAY_RATE = 0.01
REINFORCEMENT = 1
NUM_AGENTS = 50
AGENT_STEPS = 20
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# === Simple Obstacle Example ===
grid[3][4] = np.inf
grid[2][2] = 10

# === Agent Logic ===
def euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def move_agent(pos):
    x, y = pos
    candidates = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if (
            0 <= nx < GRID_SIZE and 
            0 <= ny < GRID_SIZE and 
            not np.isinf(grid[nx][ny])
        ):
            score = -grid[nx][ny] + trail_map[nx][ny] - euclidean((nx, ny), goal)
            candidates.append(((nx, ny), score))

    if not candidates:
        return pos
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# === Animation Setup ===
fig, ax = plt.subplots(figsize=(6,6))
img = ax.imshow(trail_map, cmap='plasma', origin='upper', animated=True)
plt.title("MycoNet Trail Evolution")

def update(frame):
    global trail_map
    trail_map *= (1 - DECAY_RATE)  # Decay

    for _ in range(NUM_AGENTS):
        pos = start
        for _ in range(AGENT_STEPS):
            if pos == goal:
                break
            pos = move_agent(pos)
            trail_map[pos[0]][pos[1]] += REINFORCEMENT

    img.set_array(trail_map)
    return [img]

anim = FuncAnimation(fig, update, frames=50, blit=True, interval=200)
plt.show()
