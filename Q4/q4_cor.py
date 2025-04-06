import tsplib95
import random
import math
import time
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio

def clean_tsp_file(filename, cleaned_filename="cleaned_eil76.tsp"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(cleaned_filename, 'w') as f:
        for line in lines:
            if line.startswith('BEST_KNOWN'):
                continue
            f.write(line)
    return cleaned_filename

clean_file = clean_tsp_file("eil76.tsp")
data = tsplib95.load(clean_file)
cities = list(data.get_nodes())
node_to_index = {node: i for i, node in enumerate(cities)}
coords = np.array([data.node_coords[node] for node in cities])

def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

dist_matrix = compute_distance_matrix(coords)

def get_cost(route):
    distance = 0
    for i in range(len(route)):
        from_i = node_to_index[route[i]]
        to_i = node_to_index[route[(i + 1) % len(route)]]
        distance += dist_matrix[from_i][to_i]
    return distance

def inverse(state):
    i, j = sorted(random.sample(range(len(state)), 2))
    state[i:j+1] = list(reversed(state[i:j+1]))
    return state

def insert(state):
    node = state.pop(random.randint(0, len(state) - 1))
    idx = random.randint(0, len(state))
    state.insert(idx, node)
    return state

def swap(state):
    i, j = random.sample(range(len(state)), 2)
    state[i], state[j] = state[j], state[i]
    return state

def swap_routes(state):
    a, b = sorted(random.sample(range(len(state)), 2))
    sub = state[a:b]
    del state[a:b]
    insert_pos = random.randint(0, len(state))
    for i in reversed(sub):
        state.insert(insert_pos, i)
    return state

def get_neighbors(state):
    neighbor = copy.deepcopy(state)
    random.choice([inverse, insert, swap, swap_routes])(neighbor)
    return neighbor

def annealing(initial_state, save_gif=False, gif_dir=None, max_time=600):
    initial_temp = 1000
    alpha = 0.995
    temp = initial_temp

    solution = initial_state
    best_solution = initial_state
    best_cost = get_cost(solution)

    cost_history = [best_cost]
    iteration_history = [0]
    iter_count = 0
    same_solution = 0
    same_cost_diff = 0
    run_start_time = time.time()

    gif_frame_counter = 0
    if save_gif and gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    while same_solution < 1500 and same_cost_diff < 150000:
        iter_count += 1

        if time.time() - run_start_time >= max_time:
            print("Terminating run due to time limit.")
            break

        neighbor = get_neighbors(solution)
        current_cost = get_cost(solution)
        neighbor_cost = get_cost(neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0:
            solution = neighbor
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                best_solution = neighbor
                cost_history.append(best_cost)
                iteration_history.append(iter_count)

                if save_gif and gif_dir:
                    gif_frame_counter += 1
                    plt.figure(figsize=(6, 6))
                    xs = [data.node_coords[node][0] for node in best_solution + [best_solution[0]]]
                    ys = [data.node_coords[node][1] for node in best_solution + [best_solution[0]]]
                    plt.plot(xs, ys, 'bo-', alpha=0.7)
                    plt.title(f"Frame {gif_frame_counter}, Cost: {best_cost:.2f}")
                    plt.axis('off')
                    frame_filename = os.path.join(gif_dir, f"frame_{gif_frame_counter:04d}.png")
                    plt.savefig(frame_filename)
                    plt.close()
            same_solution = 0
            same_cost_diff = 0
        elif delta == 0:
            solution = neighbor
            same_cost_diff += 1
            same_solution = 0
        else:
            if random.random() < math.exp(-delta / temp):
                solution = neighbor
                same_solution = 0
                same_cost_diff = 0
            else:
                same_solution += 1
                same_cost_diff += 1

        temp *= alpha

    run_time = time.time() - run_start_time
    return best_solution, best_cost, cost_history, iteration_history, run_time

best_route_distance = []
best_route = []
convergence_time = []
all_cost_histories = []
all_iteration_histories = []

for i in range(6):
    random.seed(time.time() + i)
    if i == 0:
        save_gif_flag = True
        gif_dir = "gif_frames"
    else:
        save_gif_flag = False
        gif_dir = None

    route, dist, cost_history, iter_history, run_time = annealing(
        cities.copy(), save_gif=save_gif_flag, gif_dir=gif_dir, max_time=600
    )

    best_route_distance.append(dist)
    best_route.append(route)
    convergence_time.append(run_time)
    all_cost_histories.append(cost_history)
    all_iteration_histories.append(iter_history)

    if i != 0:
        xs = [data.node_coords[node][0] for node in route + [route[0]]]
        ys = [data.node_coords[node][1] for node in route + [route[0]]]
        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys, 'bo-', alpha=0.7)
        plt.title(f"Run {i+1} - Final Distance: {dist:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

if os.path.exists("gif_frames"):
    frame_files = sorted([f for f in os.listdir("gif_frames") if f.endswith('.png')])
    frames = [imageio.imread(os.path.join("gif_frames", f)) for f in frame_files]
    imageio.mimsave('simulated_annealing1.gif', frames, duration=0.5)
    print("GIF saved as simulated_annealing1.gif")
    for f in frame_files:
        os.remove(os.path.join("gif_frames", f))
    os.rmdir("gif_frames")

runs = list(range(2, 7))  
times_excl_run1 = convergence_time[1:]
avg_time = np.mean(times_excl_run1)

plt.figure(figsize=(8, 6))
plt.plot(runs, times_excl_run1, marker='s', linestyle='--', color='blue', label='Run Time (Excl. Run 1)')
plt.axhline(avg_time, color='orange', linestyle='-.', label=f'Average Time: {avg_time:.2f}s')
plt.xlabel('Run Number')
plt.ylabel('Time Taken Time (seconds)')
plt.title('Time per Run (Excl. Run 1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time_taken_plot.png")
plt.close()
print(" time plot saved as 'time_taken_plot.png'")

print("\n=== Summary ===")
for i, dist in enumerate(best_route_distance):
    print(f"Run {i+1}: Distance = {dist:.2f}, Time = {convergence_time[i]:.2f}s")

best_overall_dist_excl_run1 = min(best_route_distance[1:])
print(f"\nBest Overall Distance (Excl. Run 1): {best_overall_dist_excl_run1:.2f}")
print(f"Average Time Taken (Excl. Run 1): {avg_time:.2f}s")
