import numpy as np
import matplotlib.pyplot as plt


class MapUtils:

    """Implements a map object, for easily accessing functionality"""

    def __init__(self, free=0, uncertain=-1, occupied=100):
        self.FREE = free
        self.UNCERTAIN = uncertain
        self.OCCUPIED = occupied

        self.map_origin = np.zeros((2,))
        self.occ_grid = None
        self.resolution = None

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            x, y = key
            return self.occ_grid[y, x]  # Note: numpy indexing is (row, col)
        else:
            raise IndexError("Index must be a tuple (x, y)")

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            x, y = key
            self.occ_grid[y, x] = value  # Note: numpy indexing is (row, col)
        else:
            raise IndexError("Index must be a tuple (x, y)")

    @property
    def shape(self):
        sh = self.occ_grid.shape
        return sh[1], sh[0]

    def set_map(self, occ_grid, origin, resolution):
        self.occ_grid = occ_grid
        self.map_origin = origin
        self.resolution = resolution

    def set_origin(self, origin):
        self.map_origin = origin

    def pose_to_map(self, pose):
        # Pose should be tuple (for a single pose), list of lists, or np.ndarray
        if (isinstance(pose, tuple) and len(pose) == 2) or isinstance(pose, list):
            pose = np.array(pose)
        if not isinstance(pose, np.ndarray):
            raise TypeError(f"Pose type {type(pose)} not recognized.")
        if pose.shape[-1] == 3:
            pose = pose[..., :2]
        rel_pose = pose - self.map_origin[:2]
        yaw = self.map_origin[2]
        Rinv = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        map_inds = (Rinv @ rel_pose.T / self.resolution).astype(int).T
        return map_inds[..., 0].tolist(), map_inds[..., 1].tolist()

    def map_to_pose(self, map_inds):
        if isinstance(map_inds, list):
            map_inds = np.array(map_inds)
        if not isinstance(map_inds, np.ndarray):
            raise TypeError(f"Indices type {type(map_inds)} not recognized.")
        yaw = self.map_origin[2]
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        rel_pos = (map_inds + 0.5) * self.resolution

        return self.map_origin[:2] + (R @ rel_pos.T).T

    def plot(self, ax):
        cmap = plt.cm.gray_r
        cmap.set_bad(color='gray')  # Set unknown (-1) cells to gray
        ax.imshow(np.ma.masked_where(self.occ_grid == self.UNCERTAIN, self.occ_grid), cmap=cmap, origin="lower")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle='--', alpha=0.25)