import numpy as np
import heapq
import random

class Frontier:

    def __init__(
        self, min_frontier_size,
            front_size_weight=1, front_to_goal_weight=10, start_to_front_weight=1,
        astar_depth_limit=1000000, frontier_depth_limit=10000,
        free=0, uncertain=1, occupied=2
    ):
        # Store cells
        self.map = None
        self.FREE = free
        self.UNCERTAIN = uncertain
        self.OCCUPIED = occupied
        self.FRONTIER_OPEN = 0
        self.FRONTIER_CLOSE = 1
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.all_directions = self.directions + [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # Add diagonals

        self.min_frontier_size = min_frontier_size
        self.astar_depth_limit = astar_depth_limit
        self.frontier_depth_limit = frontier_depth_limit
        self.start_to_front_weight = start_to_front_weight
        self.front_to_goal_weight = front_to_goal_weight
        self.front_size_weight = front_size_weight

    def update_map(self, map_update, origin):
        if self.map is None:
            self.map = map_update
        else:
            self.map[origin[0]:origin[0] + map_update.shape[0], origin[1]:origin[1] + map_update.shape[1]] = map_update

    @staticmethod
    def heuristic(a, b):
        return np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node, all_dir=False):
        neighbors = []
        rows, cols = self.map.shape
        directions = self.all_directions if all_dir else self.directions
        random.shuffle(directions)
        for dr, dc in directions:
            r, c = node[0] + dr, node[1] + dc
            if 0 <= r < rows and 0 <= c < cols:
                if self.map[r, c] == self.FREE:  # Check bounds and obstacles
                    neighbors.append(((r, c), self.FREE))
                elif self.map[r, c] == self.UNCERTAIN:
                    neighbors.append(((r, c), self.UNCERTAIN))
        return neighbors

    def find_frontiers_to_goal(self, start_node, goal_node, find_frontiers=True):
        assert self.map[start_node] == self.FREE
        frontier_closed_list = []       # Mark cells which have been explored in frontier exploration
        frontiers = []                  # Store all frontiers

        open_set = []                   # Set of all nodes to be explored by astar
        heapq.heappush(open_set, (0, start_node, self.FREE))
        came_from = {}                  # Linked list storing shortest path to each node
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
            elif curr_occ == self.FREE:
                # Get neighbors of the node
                neighbors = self.get_neighbors(curr_node)
                is_frontier = False
                for neighbor, occ in neighbors:
                    # Compute candidate stage cost for next node
                    tentative_g_score = g_score[curr_node] + 1

                    # If node is unexplored, or shorter path found, and node free, queue it
                    if (neighbor not in g_score or tentative_g_score < g_score[neighbor]) and occ == self.FREE:
                        came_from[neighbor] = curr_node
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor, occ))
                    # If node is uncertain and frontiers have not yet been explored for parent, run frontier algo
                    elif (not is_frontier) and find_frontiers and occ == self.UNCERTAIN:
                        # This is a frontier node!! Depth-limited BFS for frontier nodes
                        frontier = []
                        frontier_queue = [curr_node]
                        frontiers_expanded = 0
                        while frontier_queue and frontiers_expanded < self.frontier_depth_limit:
                            f_node = frontier_queue.pop(0)
                            frontiers_expanded += 1
                            if f_node in frontier_closed_list or self.map[f_node] != self.FREE:
                                continue
                            # get all neighbors
                            neighbors = self.get_neighbors(f_node, all_dir=True)
                            # Add f_node to frontier if it has a free neighbor
                            added_frontier = False
                            for i in range(len(neighbors)):
                                if neighbors[i][1] == self.UNCERTAIN:
                                    frontier.append(f_node)
                                    added_frontier = True
                                    break
                            if added_frontier:
                                for i in range(len(neighbors)):
                                    if neighbors[i][0] in frontier_queue \
                                            or neighbors[i][0]in frontier_closed_list \
                                            or self.map[neighbors[i][0]] != self.FREE:
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
            path_to_front, cost_to_front = self.select_intermediate_goal(frontiers, start_node, goal_node)
            return path_to_front, cost_to_front, frontiers
        if find_frontiers:
            return path_to_goal, cost_to_goal, frontiers
        return path_to_goal, cost_to_goal

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
