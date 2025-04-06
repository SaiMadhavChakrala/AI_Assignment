import tsplib95
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import imageio

def load_tsp(filename):
    problem = tsplib95.load(filename)
    nodes = list(problem.get_nodes())
    coords = np.array([problem.node_coords[i] for i in nodes])
    return coords, compute_distance_matrix(coords)

def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def compute_tour_cost(tour, dist_matrix):
    return np.sum(dist_matrix[tour, np.roll(tour, -1)])

def get_best_2opt_neighbor(tour, dist_matrix):
    best_cost = compute_tour_cost(tour, dist_matrix)
    best_tour = tour.copy()
    for i in range(len(tour) - 1):
        for j in range(i + 2, len(tour)):
            if j - i == 1:
                continue
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            new_cost = compute_tour_cost(new_tour, dist_matrix)
            if new_cost < best_cost:
                return new_tour, new_cost, True
    return best_tour, best_cost, False

def save_tour_frame(tour, coords, frame_dir, frame_num, cost):
    path = tour + [tour[0]]
    plt.figure(figsize=(6, 6))
    plt.plot(coords[path, 0], coords[path, 1], 'r-')
    plt.scatter(coords[:, 0], coords[:, 1], c='blue')
    plt.title(f"Frame {frame_num} | Cost: {cost:.2f}")
    plt.axis('off')
    plt.tight_layout()
    frame_path = os.path.join(frame_dir, f"frame_{frame_num:03d}.png")
    plt.savefig(frame_path)
    plt.close()

def hill_climb(coords, dist_matrix, restarts=5, max_iter=1000, gif_dir=None):
    best_tour = None
    best_cost = float('inf')
    overall_frame_count = 0
    run_results = []

    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    for restart in range(restarts + 1):  
        run_start_time = time.time()
        tour = list(np.random.permutation(len(coords)))
        cost = compute_tour_cost(tour, dist_matrix)
        iteration_count = 0
        frame_count_run = 0

        for iteration in range(max_iter):
            new_tour, new_cost, improved = get_best_2opt_neighbor(tour, dist_matrix)
            if improved:
                tour = new_tour
                cost = new_cost
                iteration_count += 1
                if gif_dir and restart == 0:
                    save_tour_frame(tour, coords, gif_dir, overall_frame_count, cost)
                    overall_frame_count += 1
                    frame_count_run += 1
            else:
                break

        run_time = time.time() - run_start_time
        run_results.append({
            'run': restart + 1,
            'final_cost': cost,
            'iterations': iteration_count,
            'frames': frame_count_run,
            'time': run_time
        })

        print(f"Run {restart+1}: Final Cost = {cost:.2f}, Iterations = {iteration_count}, "
              f"Frames = {frame_count_run}, Time = {run_time:.2f}s")

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost, overall_frame_count, run_results

def create_gif(frame_dir, output_path="hill_climb.gif"):
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(frame_dir, f)) for f in files]
    imageio.mimsave(output_path, images, duration=0.5)
    print(f"\nGIF created with {len(images)} frames at {output_path}")

def plot_run_times_line(run_results, output_file="run_time_line_excl_run1.png"):
    runs = [res['run'] for res in run_results]
    times = [res['time'] for res in run_results]
    avg_time = sum(times) / len(times)

    plt.figure(figsize=(10, 6))
    plt.plot(runs, times, marker='o', color='blue', linestyle='-', label='Run Time')
    plt.axhline(y=avg_time, color='red', linestyle='--', label=f'Avg Time = {avg_time:.2f}s')

    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel('Time Taken (seconds)', fontsize=12)
    plt.title('Hill Climbing: Time vs Run (Excl. Run 1)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    for i, v in enumerate(times):
        plt.text(runs[i], v + 0.02, f"{v:.2f}s", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"\nSaved line plot at {output_file}")

def run(filename, restarts=5, max_iter=1000, gif_enabled=True):
    coords, dist_matrix = load_tsp(filename)
    frame_dir = "gif_frames"

    overall_start_time = time.time()
    best_tour, best_cost, frame_count, run_results = hill_climb(
        coords, dist_matrix, restarts, max_iter,
        gif_dir=frame_dir if gif_enabled else None
    )
    overall_end_time = time.time()

    print("\n=== Full Summary ===")
    for res in run_results:
        print(f"Run {res['run']}: Final Cost = {res['final_cost']:.2f}, Iterations = {res['iterations']}, "
              f"Frames = {res['frames']}, Time = {res['time']:.2f}s")

    analyzed_runs = run_results[1:]

    best_excl_run1 = min(analyzed_runs, key=lambda x: x['final_cost'])
    avg_time_excl_run1 = sum(r['time'] for r in analyzed_runs) / len(analyzed_runs)

    print("\n=== Analysis (Excl. Run 1) ===")
    print(f"Best Cost (Excl. Run 1): {best_excl_run1['final_cost']:.2f} (Run {best_excl_run1['run']})")
    print(f"Average Time (Excl. Run 1): {avg_time_excl_run1:.2f}s")
    print(f"Total Time Taken = {overall_end_time - overall_start_time:.2f}s")

    plot_run_times_line(analyzed_runs)

    if gif_enabled:
        create_gif(frame_dir)
        for f in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, f))
        os.rmdir(frame_dir)

if __name__ == "__main__":
    run("eil76.tsp", restarts=5, max_iter=1000, gif_enabled=True)
