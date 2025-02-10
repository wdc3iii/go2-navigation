import rclpy
import tf2_ros
from rclpy.node import Node

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
        self.declare_parameter('map_size', [400, 1000])  # Grid size
        self.declare_parameter('resolution', 0.05)  # Map resolution
        self.declare_parameter('scan_range', 2000.0)  # LIDAR range
        self.declare_parameter('scan_resolution', 0.1)  # Scan angle step
        self.declare_parameter('drift_rate', 0.001)  # Drift per second

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.map_update_pub = self.create_publisher(OccupancyGridUpdate, '/map_updates', 1)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 3)
        self.goal_pub = self.create_publisher(Pose2D, '/obelisk/go2/goal_pose', 1)
        
        self.sim_robot_sub = self.create_subscription(ObkFramePose, '/obelisk/go2/mocap', self.robot_sim_callback, 1)
        self.mpc_sub = self.create_subscription(Trajectory, '/obelisk/go2/dtmpc_path', self.trajectory_callback, 1)

        # TF Broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.recieved_sim_robot = False

        # TF Offsets
        self.map_to_odom_x = 0.0
        self.map_to_odom_y = 0.0
        self.map_to_odom_theta = 0.0

        # Simulated Map
        self.map_origin = np.array([-2, -2])
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
        size_x, size_y = self.get_parameter('map_size').value
        self.occ_gt = zoom(self.generate_maze(size_y // 40, size_x // 40), 40, order=0)
        self.map = np.ones_like(self.occ_gt) * UNCERTAIN
        # self.map = self.occ_gt.copy()
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
        x_c, y_c = pose_to_map(self.z[:2], self.map_origin + np.array([self.map_to_odom_x, self.map_to_odom_y]), self.map_to_odom_theta, self.get_parameter('resolution').value)
        
        x1 = max(0, x_c - 75)
        x2 = min(self.get_parameter("map_size").value[1], x_c + 75)
        y1 = max(0, y_c - 90)
        y2 = min(self.get_parameter("map_size").value[1], y_c + 90)

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

    def publish_scan(self):
        """Publish a fake LIDAR scan."""
        if not self.trajectory:
            return

        # Get robot pose
        pose_x, pose_y, theta = self.trajectory[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.trajectory)

        scan_msg = LaserScan()
        scan_msg.header.frame_id = "base_link"
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.angle_min = -math.pi
        scan_msg.angle_max = math.pi
        scan_msg.angle_increment = self.get_parameter('scan_resolution').value
        scan_msg.range_min = 0.1
        scan_msg.range_max = self.get_parameter('scan_range').value

        num_readings = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment)
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
        if not self.recieved_sim_robot:
            self.sense()

    def robot_sim_callback(self, msg):
        self.z_robot = np.array([
            msg.position.x,
            msg.position.y,
            quat2yaw([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        ])
        self.recieved_sim_robot = True

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

        # Publish odom->base_link (True pose)
        if not self.recieved_sim_robot:
            tf_odom_base = TransformStamped()
            tf_odom_base.header.stamp = now
            tf_odom_base.header.frame_id = "odom"
            tf_odom_base.child_frame_id = "base_link"
            tf_odom_base.transform.translation.x = self.z[0]
            tf_odom_base.transform.translation.y = self.z[1]
            tf_odom_base.transform.rotation.z = math.sin(self.z[2] / 2)
            tf_odom_base.transform.rotation.w = math.cos(self.z[2] / 2)

            self.tf_broadcaster.sendTransform([tf_map_odom, tf_odom_base])
        elif self.recieved_sim_robot:
            tf_odom_base = TransformStamped()
            tf_odom_base.header.stamp = now
            tf_odom_base.header.frame_id = "odom"
            tf_odom_base.child_frame_id = "base_link"
            tf_odom_base.transform.translation.x = self.z_robot[0]
            tf_odom_base.transform.translation.y = self.z_robot[1]
            tf_odom_base.transform.rotation.z = math.sin(self.z_robot[2] / 2)
            tf_odom_base.transform.rotation.w = math.cos(self.z_robot[2] / 2)
            self.tf_broadcaster.sendTransform([tf_map_odom, tf_odom_base])
        else:
            self.tf_broadcaster.sendTransform([tf_map_odom])

    def publish_goal_pose(self):
        size_y, size_x = self.get_parameter('map_size').value
        resolution = self.get_parameter('resolution').value
        msg = Pose2D()
        msg.x = size_x * resolution - 4
        msg.y = size_y * resolution - 4
        msg.theta = np.pi / 4
        self.goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeSLAMNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
