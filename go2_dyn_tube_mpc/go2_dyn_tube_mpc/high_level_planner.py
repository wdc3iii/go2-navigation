from go2_dyn_tube_mpc.map_utils import MapUtils

import heapq
import random
import numpy as np
import threading
from scipy.ndimage import binary_dilation


class HighLevelPlanner:

    PLAN_TO_GOAL_FOUND = 1
    PLAN_TO_FRONTIER_FOUND = 0
    PATH_START_NOT_FREE = -1
    NO_PLAN_FOUND = -2

    def __init__(
            self, map_lock, inflated_map_lock, min_frontier_size=6, robot_radius=3,
            front_size_weight=1, front_to_goal_weight=2, start_to_front_weight=1,
            astar_depth_limit=1000000, frontier_depth_limit=10000,
            free=0, uncertain=-1, occupied=100, downsample=4
    ):
        # Store cells
        self.map_lock = map_lock
        self.inflated_map_lock = inflated_map_lock
        self.search_lock = threading.Lock()
        self.map = MapUtils(self.map_lock, free=free, uncertain=uncertain, occupied=occupied)
        self.inflated_map = MapUtils(self.inflated_map_lock, free=free, uncertain=uncertain, occupied=occupied)
        self.search_map = MapUtils(self.search_lock, free=free, uncertain=uncertain, occupied=occupied)
        self.sm_up_x1 = np.inf
        self.sm_up_x2 = -np.inf
        self.sm_up_y1 = np.inf
        self.sm_up_y2 = -np.inf
        self.robot_radius = robot_radius
        y, x = np.ogrid[-robot_radius:robot_radius + 1, -robot_radius:robot_radius + 1]
        mask = x ** 2 + y ** 2 <= robot_radius ** 2  # Circle equation
        self.kernel = mask.astype(np.uint8)  # Convert to binary mask
        # self.kernel = np.ones((2 * self.robot_radius + 1, 2 * self.robot_radius + 1))
        self.FRONTIER_OPEN = 0
        self.FRONTIER_CLOSE = 1
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.all_directions = self.directions + [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Add diagonals
        self.downsample = downsample

        self.min_frontier_size = min_frontier_size
        self.astar_depth_limit = astar_depth_limit
        self.frontier_depth_limit = frontier_depth_limit
        self.start_to_front_weight = start_to_front_weight
        self.front_to_goal_weight = front_to_goal_weight
        self.front_size_weight = front_size_weight

    def set_map(self, occ_grid, map_origin, resolution):
        # Set the nominal map
        self.map.set_map(occ_grid, map_origin, resolution)
        # Compute the inflated map over the whole space
        self.inflated_map.set_map(np.ones_like(occ_grid) * self.map.FREE, map_origin, resolution)
        self.inflate_map((0, 0), self.map.shape)
        if self.search_map.occ_grid is None:
            self.search_map.set_map(self.inflated_map.occ_grid.copy(), map_origin, resolution)
            self.sm_up_x1 = np.inf
            self.sm_up_x2 = -np.inf
            self.sm_up_y1 = np.inf
            self.sm_up_y2 = -np.inf
        else:
            self.sm_up_x1 = 0
            self.sm_up_x2 = self.map.shape[0]
            self.sm_up_y1 = 0
            self.sm_up_y2 = self.map.shape[1]

    def update_origin(self, origin):
        self.map.set_origin(origin)
        self.inflated_map.set_origin(origin)

    def update_map(self, occ_grid, origin):
        x1 = origin[0]
        x2 = origin[0] + occ_grid.shape[1]
        y1 = origin[1]
        y2 = origin[1] + occ_grid.shape[0]
        self.map[x1:x2, y1:y2] = occ_grid
        self.inflate_map(origin, occ_grid.T.shape)

    def inflate_map(self, origin, shape):
        x1 = max(origin[0] - (1 + self.robot_radius) * 2, 0)
        x2 = min(origin[0] + shape[0] + (1 + self.robot_radius) * 2, self.map.shape[0])
        y1 = max(origin[1] - (1 + self.robot_radius) * 2, 0)
        y2 = min(origin[1] + shape[1] + (1 + self.robot_radius) * 2, self.map.shape[1])
        self.sm_up_x1 = min(self.sm_up_x1, x1)
        self.sm_up_x2 = max(self.sm_up_x2, x2)
        self.sm_up_y1 = min(self.sm_up_y1, y1)
        self.sm_up_y2 = max(self.sm_up_y2, y2)

        local_map = self.map[x1:x2, y1:y2].copy()

        # Inflate uncertain
        mask_uncertain = local_map == self.map.UNCERTAIN
        inflated_mask_uncertain = binary_dilation(mask_uncertain, self.kernel)
        inflated_uncertain_local = np.where(inflated_mask_uncertain, self.map.UNCERTAIN, local_map)
        # Inflate occupied
        mask_occupied = local_map == self.map.OCCUPIED
        inflated_mask_occupied = binary_dilation(mask_occupied, self.kernel)

        self.inflated_map[x1:x2, y1:y2] = np.where(inflated_mask_occupied, self.map.OCCUPIED, inflated_uncertain_local)

    def compute_nearest_inds(self, center, size, free=True):
        return self.map.get_nearest_inds(center, size, free=free)

    @staticmethod
    def heuristic(a, b):
        return np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node, all_dir=False):
        neighbors = []
        rows, cols = self.search_map.shape
        directions = self.all_directions if all_dir else self.directions
        random.shuffle(directions)
        for dr, dc in directions:
            r, c = node[0] + dr, node[1] + dc
            if 0 <= r < rows and 0 <= c < cols:
                if self.search_map[r, c] == self.search_map.FREE:  # Check bounds and obstacles
                    neighbors.append(((r, c), self.search_map.FREE))
                elif self.search_map[r, c] == self.search_map.UNCERTAIN:
                    neighbors.append(((r, c), self.search_map.UNCERTAIN))
        return neighbors

    def find_frontiers_to_goal(self, start_pose, goal_pose, find_frontiers=True):
        if isinstance(start_pose, tuple):
            start_node = start_pose
        else:
            start_node = self.map.pose_to_map(start_pose)
        if isinstance(goal_pose, tuple):
            goal_node = goal_pose
        else:
            goal_node = self.map.pose_to_map(goal_pose)
        
        if not np.isinf(self.sm_up_x1):
            self.search_map[self.sm_up_x1:self.sm_up_x2, self.sm_up_y1:self.sm_up_y2] = self.inflated_map[self.sm_up_x1:self.sm_up_x2, self.sm_up_y1:self.sm_up_y2]
            self.sm_up_x1 = np.inf
            self.sm_up_x2 = -np.inf
            self.sm_up_y1 = np.inf
            self.sm_up_y2 = -np.inf
        # TODO: Debugging
        # print(f"Start Node: {start_node}: {self.search_map[start_node]}")
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # p = self.search_map.map_to_pose([start_node])[0]
        # self.search_map.plot(ax)
        # ax.plot(p[0], p[1], '.r')
        # ax.set_xlim([p[0] - 3, p[0] + 3])
        # ax.set_ylim([p[1] - 3, p[1] + 3])
        # ax.set_title("Beginning Inflated")
        # plt.show()
        # fig, ax = plt.subplots()
        # self.map.plot(ax)
        # ax.plot(p[0], p[1], '.r')
        # ax.set_xlim([p[0] - 3, p[0] + 3])
        # ax.set_ylim([p[1] - 3, p[1] + 3])
        # ax.set_title("Beginning Actual")
        # plt.show()

        if self.search_map[start_node] != self.search_map.FREE:
            return None, None, None, self.PATH_START_NOT_FREE
        frontier_closed_list = []       # Mark cells which have been explored in frontier exploration
        frontiers = []                  # Store all frontiers

        open_set = []                   # Set of all nodes to be explored by astar
        heapq.heappush(open_set, (0, start_node, self.search_map.FREE))
        came_from = {}                  # Linked list storing the shortest path to each node
        g_score = {start_node: 0}       # Stage cost
        f_score = {start_node: self.heuristic(start_node, goal_node)}  # Heuristic cost to go
        path_to_goal = None
        cost_to_goal = None

        nodes_expanded = 0
        while open_set and nodes_expanded < self.astar_depth_limit:
            # Pop a node
            cost_to_node, curr_node, curr_occ = heapq.heappop(open_set)
            nodes_expanded += 1

            # If the node is goal node, return path
            if curr_node == goal_node:
                path = []
                while curr_node in came_from:
                    path.append(curr_node)
                    curr_node = came_from[curr_node]
                path.append(start_node)
                path_to_goal = path[::-1]
                cost_to_goal = cost_to_node
                break

            # Continue Astar searching
            elif curr_occ == self.search_map.FREE:
                # Get neighbors of the node
                neighbors = self.get_neighbors(curr_node)
                is_frontier = False
                for neighbor, occ in neighbors:
                    # Compute candidate stage cost for next node
                    tentative_g_score = g_score[curr_node] + 1

                    # If node is unexplored, or shorter path found, and node free, queue it
                    if (neighbor not in g_score or tentative_g_score < g_score[neighbor]) and occ == self.search_map.FREE:
                        came_from[neighbor] = curr_node
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor, occ))
                    # If node is uncertain and frontiers have not yet been explored for parent, run frontier algo
                    elif (not is_frontier) and find_frontiers and occ == self.search_map.UNCERTAIN:
                        # This is a frontier node!! Depth-limited BFS for frontier nodes
                        frontier = []
                        frontier_queue = [curr_node]
                        frontiers_expanded = 0
                        while frontier_queue and frontiers_expanded < self.frontier_depth_limit:
                            f_node = frontier_queue.pop(0)
                            frontiers_expanded += 1
                            if f_node in frontier_closed_list or self.search_map[f_node] != self.search_map.FREE:
                                continue
                            # get all neighbors
                            neighbors = self.get_neighbors(f_node, all_dir=False)  # Can lead to feasibility issues if True
                            # Add f_node to frontier if it has a free neighbor
                            added_frontier = False
                            for i in range(len(neighbors)):
                                if neighbors[i][1] == self.search_map.UNCERTAIN:
                                    frontier.append(f_node)
                                    added_frontier = True
                                    break
                            if added_frontier:
                                for i in range(len(neighbors)):
                                    if neighbors[i][0] in frontier_queue \
                                            or neighbors[i][0]in frontier_closed_list \
                                            or self.search_map[neighbors[i][0]] != self.search_map.FREE:
                                        continue
                                    frontier_queue.append(neighbors[i][0])
                            frontier_closed_list.append(f_node)
                        if frontiers_expanded >= self.frontier_depth_limit:
                            print("Warning: frontier depth limit exceeded")
                        # Add the list of frontier nodes to the frontiers list.
                        if len(frontier) >= self.min_frontier_size:
                            frontiers.append((frontier, cost_to_node))  # Use cost to reach this node as rough approx for cost to reach frontier
        if nodes_expanded >= self.astar_depth_limit:
            print("Warning: A* search depth limit exceeded")
        if path_to_goal is None:
            path_to_front, cost_to_front, _, _ = self.select_intermediate_goal(frontiers, start_node, goal_node)
            return path_to_front, cost_to_front, frontiers, self.PLAN_TO_FRONTIER_FOUND

        # TODO: Debugging
        # print(f"\tEnd Node: {path_to_goal[-1]}: {self.search_map[path_to_goal[-1]]}")
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # p = self.search_map.map_to_pose([path_to_goal[-1]])[0]
        # self.search_map.plot(ax)
        # ax.plot(p[0], p[1], '.b')
        # ax.set_xlim([p[0] - 3, p[0] + 3])
        # ax.set_ylim([p[1] - 3, p[1] + 3])
        # ax.set_title("End Inflated")
        # plt.show()
        # fig, ax = plt.subplots()
        # self.map.plot(ax)
        # ax.plot(p[0], p[1], '.r')
        # ax.set_xlim([p[0] - 3, p[0] + 3])
        # ax.set_ylim([p[1] - 3, p[1] + 3])
        # ax.set_title("End Actual")
        # plt.show()

        # for r in range(self.search_map.shape[0]):
        #     for c in range(self.search_map.shape[1]):
        #         if self.search_map[r, c] == self.search_map.FREE:
        #             if (r, c) not in came_from.keys():
        #                 print(f'({r}, {c}) not in came from keys')
        # goal_node in came_from.keys()
        # self.search_map[goal_node]

        path_to_goal = self.map.map_to_pose(path_to_goal)
        return path_to_goal, cost_to_goal, frontiers, self.PLAN_TO_GOAL_FOUND

    def select_intermediate_goal(self, frontiers, start_node, goal_node):
        min_front_score = np.inf
        min_front = None
        for front, cost_to_front in frontiers:
            front_size = len(front)
            front_to_goal = self.heuristic(np.mean(np.array(front), axis=0), goal_node)

            front_score = self.front_size_weight * front_size \
                          + self.front_to_goal_weight * front_to_goal \
                          + self.start_to_front_weight * cost_to_front
            if front_score < min_front_score:
                min_front_score = front_score
                min_front = front
        min_front_pose = self.choose_frontier_pose(min_front)
        return self.find_frontiers_to_goal(start_node, min_front_pose, find_frontiers=False)

    def choose_frontier_pose(self, front):
        # TODO: choose pose in a more useful way
        front_np = np.array(front)
        centroid = np.mean(front_np, axis=0)
        v_c = front_np - centroid
        dists = np.sum(v_c * v_c, axis=1)
        return front[np.argmin(dists)]
