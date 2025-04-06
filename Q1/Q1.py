import random
import numpy as np
import imageio
import gymnasium as gym
import matplotlib.pyplot as plt
import time

n = 4
num_runs = 5
times = []

row = [1, 0, -1, 0]
col = [0, 1, 0, -1]

def generate_map(n):
    lake = []
    for i in range(n):
        row_cells = []
        for j in range(n):
            cell = random.choice(['F'] * 8 + ['H'] * 2)  # 80% Frozen, 20% Hole
            row_cells.append(cell)
        lake.append(row_cells)
    lake[0][0] = 'S'
    lake[n - 1][n - 1] = 'G'
    return lake

def isGoal(x, y, lake_map):
    return lake_map[x][y] == 'G'

def checkValidity(x, y, lake_map):
    return 0 <= x < n and 0 <= y < n and lake_map[x][y] != 'H'

def pos_to_state(x, y):
    return x * n + y

def h(new_x,new_y):
  return abs(n-1 - new_x)+abs(n-1 - new_y)

def run_dfbnb(lake_map, render_frames=False):
    global dis, U, frames

    dis = [[float('inf')] * n for _ in range(n)]
    dis[0][0] = 0
    U = float('inf')
    frames = []

    lake_map_bytes = [[c.encode("utf-8") for c in row] for row in lake_map]
    env = gym.make("FrozenLake-v1", desc=lake_map_bytes, is_slippery=False, render_mode="rgb_array")
    env.reset()

    def DFBnB(x, y):
        global U
        env.unwrapped.s = pos_to_state(x, y)
        if render_frames:
            frames.append(env.render())

        if isGoal(x, y, lake_map):
            if dis[x][y] < U:
                U = dis[x][y]
            return

        for i in range(4):
            new_x = x + row[i]
            new_y = y + col[i]
            if checkValidity(new_x, new_y, lake_map) and dis[new_x][new_y] > dis[x][y] + 1 and dis[x][y] + 1 + h(new_x, new_y) < U:
                dis[new_x][new_y] = dis[x][y] + 1
                DFBnB(new_x, new_y)
                if render_frames:
                    env.unwrapped.s = pos_to_state(x, y)
                    frames.append(env.render())

    
    if render_frames:
        frames.append(env.render())
        
    start = time.perf_counter_ns()
    DFBnB(0, 0)
    print("U values is ", U)  
    end = time.perf_counter_ns()
    elapsed_ms = (end - start) / 1_000_000  

 
    if render_frames:
        gif_filename = "first_run_dfbnb_frozenlake.gif"
        imageio.mimsave(gif_filename, frames, fps=2)
        print(f"\nGIF saved: {gif_filename}")

    return elapsed_ms, U != float('inf')


lake_map = generate_map(n)

for run in range(num_runs+1):
    print(f"\n=== Run {run + 1} ===")
    for row_ in lake_map:
        print(" ".join(row_))

    time_taken, success = run_dfbnb(lake_map, render_frames=(run == 0))
    status = "Reached Goal" if success else "No Path"
    print(f"Time: {time_taken:.2f} ms | {status}")
    if not run == 0: 
        times.append(time_taken)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_runs + 1), times, marker='o', linestyle='-', color='blue')
plt.title("DFBnB Execution Time Across Runs")
plt.xlabel("Run Number")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig("dfbnb_run_times.png")
plt.show()

avg_time = sum(times) / len(times)
print(f"\nAverage Time: {avg_time:.2f} ms")
