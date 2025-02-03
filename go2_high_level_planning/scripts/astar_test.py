import numpy as np
import matplotlib.pyplot as plt
import random
from go2_high_level_planning.exploration import Frontier
from scipy.ndimage import zoom
import time

FREE = 0
UNCERTAIN = 1
OCCUPIED = 2

display = True

def generate_maze(width, height):
    """Generate a maze using recursive backtracking algorithm."""
    maze = np.ones((height, width), dtype=int) * 2

    def carve_passages_from(x, y):
        """Recursive function to carve the maze."""
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 < nx < width-1 and 0 < ny < height-1 and maze[ny, nx] == OCCUPIED:
                maze[y + dy // 2, x + dx // 2] = FREE  # Remove wall
                maze[ny, nx] = FREE  # Mark passage
                carve_passages_from(nx, ny)

    # Start at a random point in the maze
    start_x, start_y = (random.randrange(1, width, 2), random.randrange(1, height, 2))
    maze[start_y, start_x] = FREE
    carve_passages_from(start_x, start_y)

    return maze

def plot_maze(maze, path, last_path, frontiers):
    """Plot the maze using matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(maze.T, cmap="gray_r", origin="lower")
    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    if path is not None:
        x = np.array([n[0] for n in path])
        y = np.array([n[1] for n in path])
        plt.plot(x, y, 'r')
    if last_path is not None:
        x = np.array([n[0] for n in last_path])
        y = np.array([n[1] for n in last_path])
        plt.plot(x, y, '--b')
    if frontiers is not None:
        for front, _ in frontiers:
            x = np.array([n[0] for n in front])
            y = np.array([n[1] for n in front])
            plt.plot(x, y, 'g.')
    plt.show()

if __name__ == '__main__':
    # Generate and plot the maze
    maze_width, maze_height = 51, 51  # Odd dimensions for proper maze generation
    scale = 4
    sensing_range = 10
    occ_grid_gt = generate_maze(maze_width, maze_height)

    occ_grid_gt = zoom(occ_grid_gt, scale, order=0)
    front = Frontier(2, free=FREE, uncertain=UNCERTAIN, occupied=OCCUPIED)
    front.update_map(np.ones_like(occ_grid_gt), (0, 0))
    start = None
    goal = None
    for r in range(maze_height * scale):
        for c in range(maze_width * scale):
            if occ_grid_gt[r][c] == FREE:
                if start is None:
                    start = (r, c)
                goal = (r, c)

    total_path = []
    last_path = None
    path = None
    solved = False
    curr_state = start
    plot_maze(occ_grid_gt, [start], [goal], None)

    t0 = time.perf_counter()
    while not solved:
        last_path = path
        # Sense
        r_l = max(curr_state[0] - sensing_range, 0)
        r_u = min(curr_state[0] + sensing_range, maze_height * scale)
        c_l = max(curr_state[1] - sensing_range, 0)
        c_u = min(curr_state[1] + sensing_range, maze_height * scale)

        t1 = time.perf_counter()
        front.update_map(occ_grid_gt[r_l:r_u, c_l:c_u], (r_l, c_l))
        # Plan
        path, cost, frontiers = front.find_frontiers_to_goal(curr_state, goal)
        print(cost, time.perf_counter() - t1)

        # Display the maze
        if display:
            plot_maze(front.map, path, last_path, frontiers)

        curr_state = path[-1]
        if path[-1] == goal:
            solved = True
        total_path.extend(path)

    print(time.perf_counter() - t0)
    plot_maze(front.map, total_path, None, None)
