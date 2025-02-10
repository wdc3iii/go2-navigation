import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from obelisk_sensor_msgs.msg import ObkFramePose
from go2_dyn_tube_mpc_msgs.msg import Trajectory
from geometry_msgs.msg import TransformStamped, Pose2D

from go2_dyn_tube_mpc.map_utils import pose_to_map

import math
import random
import numpy as np
from scipy.ndimage import zoom

FREE = 0
UNCERTAIN = -1
OCCUPIED = 100


def quat2yaw(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw


class FakeSLAMNode(Node):
    def __init__(self):
        super().__init__('fake_slam')

        # Parameters
        self.declare_parameter('problem_ind', 1)  # Whether to listen to robot simulation or just execute mpc.
        prob_ind = self.get_parameter('problem_ind').value
        if prob_ind == 0:
            map_size = [400, 1000]
            sense_size = [75, 90]
        elif prob_ind == 1 or prob_ind == 2:
            map_size = [60, 60]
            sense_size = [15, 15]
            scan_range = 2000.0 if prob_ind == 1 else 3.5
        else:
            raise ValueError(f"ROS2 parameter problem_ind {prob_ind} unrecognized.")
        self.declare_parameter('map_size', map_size)  # Grid size
        self.declare_parameter('sense_size', sense_size)  # Grid size
        self.declare_parameter('resolution', 0.05)  # Map resolution
        self.declare_parameter('scan_range', scan_range)  # LIDAR range
        self.declare_parameter('scan_resolution', 0.1)  # Scan angle step
        self.declare_parameter('drift_rate', 0.001)  # Drift per second
        self.declare_parameter('use_robot_sim', False)  # Whether to listen to robot simulation or just execute mpc.
        

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.map_update_pub = self.create_publisher(OccupancyGridUpdate, '/map_updates', 1)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 3)
        self.goal_pub = self.create_publisher(Pose2D, '/obelisk/go2/goal_pose', 1)
        
        self.sim_robot_sub = self.create_subscription(ObkFramePose, '/obelisk/go2/mocap', self.robot_sim_callback, 1)
        self.mpc_sub = self.create_subscription(Trajectory, '/obelisk/go2/dtmpc_path', self.trajectory_callback, 1)

        # TF Broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # TF Offsets
        self.map_to_odom_x = 0.0
        self.map_to_odom_y = 0.0
        self.map_to_odom_theta = 0.0

        # Simulated Map
        self.z = np.zeros((3,))
        self.z_robot = np.zeros((3,))
        self.generate_fake_map()

        # Ground Truth Trajectory
        self.trajectory = []
        self.current_index = 0

        # Timers
        self.create_timer(1.0, self.publish_map)
        self.create_timer(1.0, self.publish_goal_pose)
        self.create_timer(0.1, self.publish_scan)
        self.create_timer(0.1, self.publish_tf)

    def generate_maze(self, width, height):
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

    def generate_fake_map(self):
        """Generate a simple occupancy grid."""
        problem_ind = self.get_parameter("problem_ind").value
        if problem_ind == 0:
            size_x, size_y = self.get_parameter('map_size').value
            self.occ_gt = zoom(self.generate_maze(size_y // 40, size_x // 40), 40, order=0)
            self.map = np.ones_like(self.occ_gt) * UNCERTAIN
            self.map_origin = np.array([-2, -2])

            resolution = self.get_parameter('resolution').value
            self.goal_pose = np.array([size_x * resolution - 2, size_y * resolution - 2, np.pi / 4])

        elif problem_ind == 1 or problem_ind == 2:
            self.set_parameters([Parameter('map_size'), Parameter.Type.INTEGER_ARRAY, [60, 60]])
            self.occ_gt = np.ones((60, 60)) * FREE
            self.occ_gt[:, [0, -1]] = OCCUPIED
            self.occ_gt[[0, -1], :] = OCCUPIED
            self.map_origin = np.array([-0.75, -0.75])
            self.goal_pose = np.array([1.25, 1.25, np.pi / 2])
            if problem_ind == 1:
                self.occ_gt[24:28, 24:28] = OCCUPIED
                self.map = np.ones_like(self.occ_gt) * UNCERTAIN
            else:
                self.map = self.occ_gt.copy()

        else:
            raise ValueError(f'ROS2 parameter problem_ind {problem_ind} not recognized.')
        self.sense()

    def publish_map(self):
        """Publish the full map periodically."""
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = "map"
        size_y, size_x = self.get_parameter('map_size').value
        resolution = self.get_parameter('resolution').value
        map_msg.info.width = size_x
        map_msg.info.height = size_y
        map_msg.info.resolution = resolution
        map_msg.info.origin.position.x = self.map_origin[0] + self.map_to_odom_x
        map_msg.info.origin.position.y = self.map_origin[1] + self.map_to_odom_y
        map_msg.info.origin.orientation.w = np.cos(self.map_to_odom_theta / 2)
        map_msg.info.origin.orientation.z = np.sin(self.map_to_odom_theta / 2)
        map_msg.data = self.map.flatten().astype(int).tolist()
        self.map_pub.publish(map_msg)

    def sense(self):
        z_sense = None
        if self.get_parameter('use_robot_sim').value:
            z_sense = self.z_robot
        else:
            z_sense = self.z
        x_c, y_c = pose_to_map(z_sense[:2], self.map_origin + np.array([self.map_to_odom_x, self.map_to_odom_y]), self.map_to_odom_theta, self.get_parameter('resolution').value)
        
        sense_x, sense_y = self.get_parameter('sense_size').value
        x1 = max(0, x_c - sense_x)
        x2 = min(self.get_parameter("map_size").value[1], x_c + sense_x)
        y1 = max(0, y_c - sense_y)
        y2 = min(self.get_parameter("map_size").value[1], y_c + sense_y)

        update = OccupancyGridUpdate()
        update.header.stamp = self.get_clock().now().to_msg()
        update.header.frame_id = "map"
        update.width = int(x2 - x1)
        update.height = int(y2 - y1)
        update.x = int(x1)
        update.y = int(y1)
        
        # Generate a random update in this region
        update.data = self.occ_gt[y1:y2, x1:x2].flatten().astype(int).tolist()
        self.map[y1:y2, x1:x2] = self.occ_gt[y1:y2, x1:x2]

        self.map_update_pub.publish(update)

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

    def publish_scan(self):
        """Publish a fake LIDAR scan."""
        # Get robot pose
        scan_msg = LaserScan()
        scan_msg.header.frame_id = "base_link"
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.angle_min = -math.pi
        scan_msg.angle_max = math.pi
        scan_msg.angle_increment = self.get_parameter('scan_resolution').value
        scan_msg.range_min = 0.1
        scan_msg.range_max = self.get_parameter('scan_range').value

        num_readings = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment)
        

        prob_ind = self.get_parameter('prob_ind').value
        
        if prob_ind == 2:
            z_sense = None
            if self.get_parameter('use_robot_sim').value:
                z_sense = self.z_robot
            else:
                z_sense = self.z
            obs = [
                (1.3, 1.3, 1.3, 1.8),
                (1.3, 1.8, 1.8, 1.8),
                (1.8, 1.8, 1.8, 1.3),
                (1.8, 1.3, 1.3, 1.3)
            ]
            ranges = []
            for angle in np.linspace(scan_msg.angle_min, scan_msg.angle_max, num_readings):
                total_angle = angle + z_sense[2]
                min_ray = np.inf
                for x1, x2, y1, y2 in obs:
                    ray = self.ray_intersect_segment(np.array([x1, y1]), np.array([x2, y2]), np.array([np.cos(total_angle), np.sin(total_angle)]), z_sense[:2])
                    if ray is not None and ray < min_ray:
                        min_ray = ray
                ranges.append(min(min_ray, scan_msg.range_max))

            scan_msg.ranges = ranges 
        else:
            scan_msg.ranges = [scan_msg.range_max / 2] * num_readings  # Simulated LIDAR scan
        self.scan_pub.publish(scan_msg)

    def trajectory_callback(self, msg):
        v = np.array(msg.v).reshape(msg.horizon, msg.m)
        v0 = v[0]
        t = msg.t

        self.z = self.z + np.array([
            v0[0] * np.cos(self.z[2]) - v0[1] * np.sin(self.z[2]),
            v0[0] * np.sin(self.z[2]) + v0[1] * np.cos(self.z[2]),
            v0[2]
        ]) * (t[1] - t[0])
        self.sense()

    def robot_sim_callback(self, msg):
        self.z_robot = np.array([
            msg.position.x,
            msg.position.y,
            quat2yaw([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        ])

    def publish_tf(self):
        """Publish TF transforms with drift in map->odom."""
        now = self.get_clock().now().to_msg()

        # Introduce Drift in map -> odom
        drift_rate = self.get_parameter('drift_rate').value
        self.map_to_odom_x += drift_rate * 0.
        self.map_to_odom_y += drift_rate * 0.
        self.map_to_odom_theta += drift_rate * 0.

        tf_map_odom = TransformStamped()
        tf_map_odom.header.stamp = now
        tf_map_odom.header.frame_id = "map"
        tf_map_odom.child_frame_id = "odom"
        tf_map_odom.transform.translation.x = self.map_to_odom_x
        tf_map_odom.transform.translation.y = self.map_to_odom_y
        tf_map_odom.transform.rotation.z = math.sin(self.map_to_odom_theta / 2)
        tf_map_odom.transform.rotation.w = math.cos(self.map_to_odom_theta / 2)

        tf_odom_base = TransformStamped()
        tf_odom_base.header.stamp = now
        tf_odom_base.header.frame_id = "odom"
        tf_odom_base.child_frame_id = "base_link"
        
        # Publish odom->base_link (True pose)
        if self.get_parameter('use_robot_sim').value:
            tf_odom_base.transform.translation.x = self.z_robot[0]
            tf_odom_base.transform.translation.y = self.z_robot[1]
            tf_odom_base.transform.rotation.z = math.sin(self.z_robot[2] / 2)
            tf_odom_base.transform.rotation.w = math.cos(self.z_robot[2] / 2)
        else:            
            tf_odom_base.transform.translation.x = self.z[0]
            tf_odom_base.transform.translation.y = self.z[1]
            tf_odom_base.transform.rotation.z = math.sin(self.z[2] / 2)
            tf_odom_base.transform.rotation.w = math.cos(self.z[2] / 2)
        
        self.tf_broadcaster.sendTransform([tf_map_odom, tf_odom_base])

    def publish_goal_pose(self):
        msg = Pose2D()
        msg.x = self.goal_pose[0]
        msg.y = self.goal_pose[1]
        msg.theta = self.goal_pose[2]
        self.goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeSLAMNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()