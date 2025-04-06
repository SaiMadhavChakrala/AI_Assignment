import random
import numpy as np
import imageio
import gymnasium as gym
import matplotlib.pyplot as plt
import time

n = 4
num_runs = 5
times = []

row = [-1, 1, 0, 0]
col = [0, 0, 1, -1]
U1 = 0
U = 0

def generate_map(n):
    lake = []
    for i in range(n):
        row_cells = []
        for j in range(n):
            cell = random.choice(['F'] * 8 + ['H'] * 2)
            row_cells.append(cell)
        lake.append(row_cells)
    lake[0][0] = 'S'
    lake[n - 1][n - 1] = 'G'
    return lake

def isGoal(x, y, lake_map):
    return lake_map[x][y] == 'G'

def checkValidity(x, y, lake_map=None):
  if x<0 or x>=n or y<0 or y>=n:
    return False
  if lake_map[x][y]=='H':
    return False
  return True

def pos_to_state(x, y):
    return x * n + y

def h(new_x, new_y):
    return abs(n - 1 - new_x) + abs(n - 1 - new_y)

best_dis=0

def run_idastar(lake_map, render_frames=False):
    global frames,best_dis
    frames = []

    lake_map_bytes = [[c.encode("utf-8") for c in row] for row in lake_map]
    env = gym.make("FrozenLake-v1", desc=lake_map_bytes, is_slippery=False, render_mode="rgb_array")
    env.reset()

    def IDA(x, y, dis):
        global U, U1, best_dis 
        env.unwrapped.s = pos_to_state(x, y)
        if render_frames:
            frames.append(env.render())

        if isGoal(x, y, lake_map):
            best_dis=dis
            return True

        for i in range(4):
            new_x = x + row[i]
            new_y = y + col[i]
            if checkValidity(new_x,new_y,lake_map):
                if dis+1+h(new_x,new_y)<=U :
                    if IDA(new_x,new_y,dis+1) == True :
                        return True
                else :
                    if dis+1 + h(new_x, new_y) < U1:
                        print(h(new_x, new_y))
                        print(f"{new_x} + {new_y}")
                        U1=dis+1+h(new_x,new_y)
        return False

    def IDA_star_driver():
        global U,U1
        flag=False
        U1=0
        while not flag:
            U=U1
            U1=n*n+9
            flag=IDA(0,0,0)
        return U

    if render_frames:
        frames.append(env.render())

    start = time.perf_counter_ns()
    found = IDA_star_driver()
    print(f"Answer(U,dis):- {U} , {best_dis} ")
    end = time.perf_counter_ns()
    elapsed_ms = (end - start) / 1_000_000

    if render_frames:
        gif_filename = "first_run_idastar_frozenlake.gif"
        imageio.mimsave(gif_filename, frames, fps=2)
        print(f"\nGIF saved: {gif_filename}")

    return elapsed_ms, found

lake_map = generate_map(n)

for run in range(num_runs + 1):
    print(f"\n=== Run {run + 1} ===")
    for row_ in lake_map:
        print(" ".join(row_))

    time_taken, success = run_idastar(lake_map, render_frames=(run == 0))
    status = "Reached Goal" if success else "No Path"
    print(f"Time: {time_taken:.2f} ms | {status}")
    if run != 0:
        times.append(time_taken)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_runs + 1), times, marker='o', linestyle='-', color='green')
plt.title("IDA* Execution Time Across Runs")
plt.xlabel("Run Number")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig("idastar_run_times.png")
plt.show()

avg_time = sum(times) / len(times)
print(f"\nAverage Time: {avg_time:.2f} ms")
