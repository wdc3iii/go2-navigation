import numpy as np
import matplotlib.pyplot as plt
import random
from go2_high_level_planning.exploration import Frontier
from scipy.ndimage import zoom


def generate_maze(width, height):
    """Generate a maze using recursive backtracking algorithm."""
    maze = np.ones((height, width), dtype=int) * Frontier.OCCUPIED

    def carve_passages_from(x, y):
        """Recursive function to carve the maze."""
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 < nx < width-1 and 0 < ny < height-1 and maze[ny, nx] == Frontier.OCCUPIED:
                maze[y + dy // 2, x + dx // 2] = Frontier.FREE  # Remove wall
                maze[ny, nx] = Frontier.FREE  # Mark passage
                carve_passages_from(nx, ny)

    # Start at a random point in the maze
    start_x, start_y = (random.randrange(1, width, 2), random.randrange(1, height, 2))
    maze[start_y, start_x] = 0
    carve_passages_from(start_x, start_y)

    return maze

def plot_maze(maze, path):
    """Plot the maze using matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(maze.T, cmap="binary", origin="lower")
    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    if path is not None:
        x = np.array([n[0] for n in path])
        y = np.array([n[1] for n in path])
        plt.plot(x, y, 'r')
    plt.show()

if __name__ == '__main__':
    # Generate and plot the maze
    maze_width, maze_height = 21, 21  # Odd dimensions for proper maze generation
    occupancy_grid = generate_maze(maze_width, maze_height)
    scale = 4
    occupancy_grid = zoom(occupancy_grid, scale, order=0)
    front = Frontier()
    front.update_map(occupancy_grid, (0, 0))
    start = None
    goal = None
    for r in range(maze_height * scale):
        for c in range(maze_width * scale):
            if occupancy_grid[r][c] == Frontier.FREE:
                if start is None:
                    start = (r, c)
                goal = (r, c)
    path = front.find_frontiers_to_goal(start, goal)

    # Display the maze
    plot_maze(occupancy_grid, path)


    # Return the occupancy grid
    occupancy_grid

