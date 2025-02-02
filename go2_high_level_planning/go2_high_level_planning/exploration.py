import numpy as np
import heapq
import random

class Frontier:
    FREE = 0
    UNCERTAIN = 1
    OCCUPIED = 100

    def __init__(self):
        # Store cells
        self.map = None
        self.wavefront = None
        self.MAP_OPEN = 1
        self.MAP_CLOSE = 2
        self.FRONTIER_OPEN = 3
        self.FRONTIER_CLOSE = 4
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right


        self.astar_depth_limit = 1000000
        self.frontier_depth_limit = 10000

    def update_map(self, map_update, origin):
        if self.map is None:
            self.map = map_update
        else:
            self.map[origin[0]:origin[0] + map_update.shape[0], origin[1]:origin[1] + map_update.shape[1]] = map_update

    @staticmethod
    def heuristic(a, b):
        return np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        neighbors = []
        rows, cols = self.map.shape
        random.shuffle(self.directions)
        for dr, dc in self.directions:
            r, c = node[0] + dr, node[1] + dc
            if 0 <= r < rows and 0 <= c < cols:
                if self.map[r, c] == self.FREE:  # Check bounds and obstacles
                    neighbors.append(((r, c), self.FREE))
                elif self.map[r, c] == self.UNCERTAIN:
                    neighbors.append(((r, c), self.UNCERTAIN))
        return neighbors

    def find_frontiers_to_goal(self, start_node, goal_node):
        node_markings = np.zeros_like(self.map)

        open_set = []
        heapq.heappush(open_set, (0, start_node, self.FREE))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, goal_node)}
        path_to_goal = None

        while open_set:
            _, curr_node, curr_occ = heapq.heappop(open_set)

            if curr_node == goal_node:
                path = []
                while curr_node in came_from:
                    path.append(curr_node)
                    curr_node = came_from[curr_node]
                path.append(start_node)
                path_to_goal = path[::-1]
                break

            neighbors = self.get_neighbors(curr_node)
            for neighbor, occ in neighbors:
                tentative_g_score = g_score[curr_node] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = curr_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor, occ))

        return path_to_goal



