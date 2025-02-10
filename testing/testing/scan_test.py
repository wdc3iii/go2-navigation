import numpy as np
import matplotlib.pyplot as plt

def ray_intersects_segment(dx, dy, x1, y1, x2, y2):
    """Compute the intersection of a ray (px, py) + t*(dx, dy) with a segment (x1, y1) to (x2, y2)."""
    denom = (x2 - x1) * (-dy) - (y2 - y1) * (-dx)
    if abs(denom) < 1e-6:  # Parallel lines
        return None

    t1 = ((0 - x1) * (-dy) - (0 - y1) * (-dx)) / denom
    t2 = ((x1 - 0) * (y2 - y1) - (y1 - 0) * (x2 - x1)) / denom

    if 0 <= t1 <= 1 and t2 >= 0:  # Ensure intersection is on the segment and along the ray direction
        return 0 + t2 * dx, 0 + t2 * dy

    return None

def ray_intersect_segment(p1, p2, p3, d):
    M = np.hstack([-p3[:, None], p2[:, None] - p1[:, None]])
    b = d - p1
    if np.linalg.cond(M) < 0.0001:
        return None
    else:
        alpha = np.linalg.solve(M, b)
        if alpha[0] < 0 or alpha[1] < 0 or alpha[1] > 1:
            return np.inf
        else:
            return alpha[0]

def publish_scan(z_sense):
    """Publish a fake LIDAR scan."""
    # Get robot pose
    angle_min = -np.pi
    angle_max = np.pi
    angle_increment = 0.1
    range_min = 0.1
    range_max = 3.5

    num_readings = int((angle_max - angle_min) / angle_increment)
    
    obs = [
        (1.3, 1.3, 1.3, 1.8),
        (1.3, 1.8, 1.8, 1.8),
        (1.8, 1.8, 1.8, 1.3),
        (1.8, 1.3, 1.3, 1.3)
    ]
    plt.plot(z_sense[0], z_sense[1], 'ro')
    for x1, x2, y1, y2 in obs:
        plt.plot([x1, x2], [y1, y2], 'b')
    
    ranges = []
    for angle in np.linspace(angle_min, angle_max, num_readings):
        total_angle = angle + z_sense[2]
        min_ray = np.inf
        for x1, x2, y1, y2 in obs:
            ray = ray_intersect_segment2(np.array([x1, y1]), np.array([x2, y2]), np.array([np.cos(total_angle), np.sin(total_angle)]), z_sense[:2])
            if ray is not None and ray < min_ray:
                min_ray = ray
        ranges.append(min(min_ray, range_max))
        print(total_angle, ranges[-1], np.cos(total_angle), np.sin(total_angle))
        plt.plot(z_sense[0] + ranges[-1] * np.cos(total_angle), z_sense[1] + ranges[-1] * np.sin(total_angle), 'k.')
    plt.show()

if __name__ == "__main__":
    publish_scan([0, 0, 0])
    publish_scan([0, 0, 1])
    publish_scan([0, 1, 0])
    publish_scan([1, 0, 0])
