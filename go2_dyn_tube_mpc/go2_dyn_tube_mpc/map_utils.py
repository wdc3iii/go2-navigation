import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.ndimage import distance_transform_edt


def map_to_pose(map_inds, map_origin, map_theta, resolution):
    R = np.array([
        [np.cos(map_theta), -np.sin(map_theta)],
        [np.sin(map_theta), np.cos(map_theta)]
    ])
    rel_pos = (map_inds + 0.5) * resolution

    return map_origin + (R @ rel_pos.T).T


def pose_to_map(pose, map_origin, map_theta, resolution):
    if len(pose.shape) == 2 and pose.shape[1] == 3:
        pose = pose[:, :2]
    rel_pose = pose - map_origin
    Rinv = np.array([
        [np.cos(map_theta), np.sin(map_theta)],
        [-np.sin(map_theta), np.cos(map_theta)]
    ])
    map_inds = (Rinv @ rel_pose.T / resolution).astype(int).T
    return map_inds[..., 0], map_inds[..., 1]


class MapUtils:

    """Implements a map object, for easily accessing functionality"""

    def __init__(self, lock, free=0, uncertain=-1, occupied=100):
        self.lock = lock
        self.FREE = free
        self.UNCERTAIN = uncertain
        self.OCCUPIED = occupied

        self.map_origin = np.zeros((2,))
        self.map_theta = 0
        self.occ_grid = None
        self.resolution = None

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            x, y = key
            with self.lock:
                return self.occ_grid[y, x]  # Note: numpy indexing is (row, col)
        else:
            raise IndexError("Index must be a tuple (x, y)")

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            x, y = key
            # if isinstance(value, np.ndarray):
            #     self.occ_grid[y, x] = value.T
            # else:
            #     self.occ_grid[y, x] = value  # Note: numpy indexing is (row, col)
            with self.lock:
                self.occ_grid[y, x] = value
        else:
            raise IndexError("Index must be a tuple (x, y)")

    @property
    def shape(self):
        with self.lock:
            sh = self.occ_grid.shape
        return sh[1], sh[0]

    def set_map(self, occ_grid, origin, resolution):
        with self.lock:
            self.occ_grid = occ_grid
            self.map_origin = origin[:2]
            self.map_theta = origin[2]
            self.resolution = resolution

    def set_origin(self, origin):
        with self.lock:
            self.map_origin = origin[:2]
            self.map_theta = origin[2]

    def pose_to_map(self, pose):
        # Pose should be tuple (for a single pose), list of lists, or np.ndarray
        if (isinstance(pose, tuple) and len(pose) == 2) or isinstance(pose, list):
            pose = np.array(pose)
        if not isinstance(pose, np.ndarray):
            raise TypeError(f"Pose type {type(pose)} not recognized.")
        if pose.shape[-1] == 3:
            pose = pose[..., :2]
        with self.lock:
            map_x, map_y = pose_to_map(pose, self.map_origin, self.map_theta, self.resolution)
        map_x = np.clip(map_x, 0, self.shape[0] - 1)
        map_y = np.clip(map_y, 0, self.shape[1] - 1)
        return map_x.tolist(), map_y.tolist()


    def map_to_pose(self, map_inds):
        if isinstance(map_inds, list):
            map_inds = np.array(map_inds)
        if not isinstance(map_inds, np.ndarray):
            raise TypeError(f"Indices type {type(map_inds)} not recognized.")
        with self.lock:
            return map_to_pose(map_inds, self.map_origin, self.map_theta, self.resolution)

    def get_nearest_inds(self, center, size, free=True):
        center_inds = self.pose_to_map(center)
        x1 = max(center_inds[0] - size, 0)
        x2 = min(center_inds[0] + size, self.shape[0])
        y1 = max(center_inds[1] - size // 2, 0)
        y2 = min(center_inds[1] + size // 2, self.shape[1])

        input_im = self[x1:x2, y1:y2] == self.FREE
        if not free:
            input_im = np.logical_not(input_im)
        nearest_dists, nearest_inds = distance_transform_edt(input_im, return_indices=True)
        nearest_dists *= self.resolution

        dx = x1 * self.resolution
        dy = y1 * self.resolution
        with self.lock:
            sub_map_p = self.map_origin + np.array([
                dx * np.cos(self.map_theta) - dy * np.sin(self.map_theta),
                dx * np.sin(self.map_theta) + dy * np.cos(self.map_theta),
            ])

            return np.flip(nearest_inds, axis=0).transpose(0, 2, 1), nearest_dists.T, sub_map_p, self.map_theta

    def plot(self, ax):
        # extent = np.array([-0.5, self.shape[0] - 0.5, -0.5, self.shape[1] - 0.5]) * self.resolution
        extent = np.array([-0., self.shape[0], -0., self.shape[1]]) * self.resolution

        cmap = plt.cm.gray_r
        cmap.set_bad(color='gray')  # Set unknown (-1) cells to gray
        with self.lock:
            im = ax.imshow(np.ma.masked_where(self.occ_grid == self.UNCERTAIN, self.occ_grid),
                    cmap=cmap, origin="lower", extent=extent)
            # TODO: Fix visualization under transformation
            trans = mtransforms.Affine2D().rotate_deg(self.map_theta * 180 / np.pi).translate(self.map_origin[0], self.map_origin[1]) + ax.transData
        im.set_transform(trans)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle='--', alpha=0.25)