from go2_dyn_tube_mpc.map_utils import MapUtils
import numpy as np
from scipy.ndimage import binary_dilation


class MapHandler:
    def __init__(self, robot_radius, free=0, uncertain=-1, occupied=100):
        self.map = MapUtils(free=free, uncertain=uncertain, occupied=occupied)
        self.inflated_map = MapUtils(free=free, uncertain=uncertain, occupied=occupied)
        self.robot_radius = robot_radius
        self.kernel = np.ones((2 * self.robot_radius + 1, 2 * self.robot_radius + 1))

    def set_map(self, occ_grid, map_origin, resolution):
        # Set the nominal map
        self.map.set_map(occ_grid, map_origin, resolution)
        # Compute the inflated map over the whole space
        self.inflated_map.set_map(np.ones_like(occ_grid) * self.map.FREE, map_origin, resolution)
        self.inflate_map((0, 0), self.map.shape)

    def update_origin(self, origin):
        self.map.set_origin(origin)
        self.inflated_map.set_origin(origin)

    def update_map(self, occ_grid, origin):
        x1 = origin[0]
        x2 = origin[0] + occ_grid.shape[0]
        y1 = origin[1]
        y2 = origin[1] + occ_grid.shape[1]
        self.map[x1:x2, y1:y2] = occ_grid
        self.inflate_map(origin, occ_grid.shape)

    def inflate_map(self, origin, shape):
        x1 = max(origin[0] - self.robot_radius, 0)
        x2 = min(origin[0] + shape[0] - self.robot_radius, self.map.shape[0])
        y1 = max(origin[1] - self.robot_radius, 0)
        y2 = min(origin[1] + shape[1] - self.robot_radius, self.map.shape[1])

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
