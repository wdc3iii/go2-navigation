import numpy as np
import matplotlib.pyplot as plt
import random
from go2_dyn_tube_mpc.go2_dyn_tube_mpc.high_level_planner import HighLevelPlanner
from scipy.ndimage import zoom
import time
import threading
FREE = 0
UNCERTAIN = 1
OCCUPIED = 2

display = True

def generate_maze(width, height):
    """Generate a maze using recursive backtracking algorithm."""
    maze = np.ones((height, width), dtype=int) * OCCUPIED

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

def plot_maze(front, path, last_path, frontiers):
    """Plot the maze using matplotlib."""
    fig, ax = plt.subplots()
    front.map.plot(ax=ax)
    # ax.set_xticks([]), ax.set_yticks([])  # Hide axis ticks
    if path is not None:
        x = np.array([n[0] for n in path])
        y = np.array([n[1] for n in path])
        ax.plot(x, y, 'r')
    if last_path is not None:
        x = np.array([n[0] for n in last_path])
        y = np.array([n[1] for n in last_path])
        ax.plot(x, y, '--b')
    if frontiers is not None:
        for f, _ in frontiers:
            f = front.map.map_to_pose(f)
            x = np.array([n[0] for n in f])
            y = np.array([n[1] for n in f])
            ax.plot(x, y, 'g.')
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 5])
    plt.show()

if __name__ == '__main__':
    # Generate and plot the maze
    occ_grid_gt = np.array([
        [FREE, UNCERTAIN, UNCERTAIN, UNCERTAIN, UNCERTAIN],
        [FREE, FREE, OCCUPIED, OCCUPIED, OCCUPIED],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, FREE, FREE, FREE],
        [FREE, FREE, OCCUPIED, OCCUPIED, OCCUPIED],
        [FREE, FREE, UNCERTAIN, UNCERTAIN, UNCERTAIN]
    ])

    front = HighLevelPlanner(threading.Lock(), threading.Lock(), 1, free=FREE, uncertain=UNCERTAIN, occupied=OCCUPIED, robot_radius=1)
    front.set_map(np.ones_like(occ_grid_gt) * UNCERTAIN, (0, 0, 0), 0.1)
    front.update_map(np.ones((2, 3)) * OCCUPIED, (1, 0))
    # start = None
    # goal = None
    # for r in range(maze_height * scale):
    #     for c in range(maze_width * scale):
    #         if front.inflated_map[r, c] == FREE:
    #             if start is None:
    #                 start = [r, c]
    #             goal = [r, c]
    start = [5, 5]
    goal = [199, 199]
    curr_pose = front.search_map.map_to_pose(start)
    goal_pose = front.search_map.map_to_pose(goal)
    total_path = []
    last_path = None
    path = None
    solved = False
    plot_maze(front, [curr_pose], [goal_pose], None)
    curr_state = start

    t0 = time.perf_counter()
    while not solved:
        last_path = path
        # Sense
        r_l = max(curr_state[0] - sensing_range, 0)
        r_u = min(curr_state[0] + sensing_range, maze_height * scale)
        c_l = max(curr_state[1] - sensing_range, 0)
        c_u = min(curr_state[1] + sensing_range, maze_height * scale)

        t1 = time.perf_counter()
        front.update_map(occ_grid_gt[c_l:c_u, r_l:r_u], (r_l, c_l))
        # Plan
        path, cost, frontiers = front.find_frontiers_to_goal(curr_pose, goal_pose)
        print(cost, time.perf_counter() - t1)

        # Display the maze
        if display:
            plot_maze(front, path, last_path, frontiers)

        curr_pose = path[-1]
        curr_state = front.search_map.pose_to_map(curr_pose)
        if np.all(curr_pose == goal_pose):
            solved = True
        total_path.extend(path)

    print(time.perf_counter() - t0)
    plot_maze(front, total_path, None, None)
