import numpy as np

class HighLevelPlanner:

    def __init__(self):
        # Current goal pose, current pose
        self.goal_pose = np.zeros((3,))
        self.pose = np.zeros((3,))

        # Intermediate goal pose
        self.intermediate_goal = None

        # Mapping parameters
        self.d_width: float = 0.
        self.d_height: float = 0.
        self.width: int = 0
        self.height: float  = 0

        self.occ_grid = None    # Want occ grid for planning via A*
        self.uncertain = None   # Want uncertain list
        self.occupied = None    #

    def set_map(self, occ_grid):
        self.occ_grid = occ_grid

    def update_map(self, update_map):
        # TODO: update map
        pass

    def set_goal_pose(self, goal_pose):
        assert goal_pose.shape == self.goal_pose.shape
        self.goal_pose = goal_pose

    def set_pose(self, pose):
        assert pose.shape == self.pose.shape
        self.pose = pose

    def navigate_to_goal(self):
        goal = self.intermediate_goal if self.intermediate_goal else self.goal_pose

        # Run A* from current pose to goal pose

        return path
