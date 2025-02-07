import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
import tf2_ros
import math

from geometry_msgs.msg import TransformStamped, Pose2D
from nav_msgs.msg import OccupancyGrid, MapMetaData
from map_msgs.msg import OccupancyGridUpdate  # Import the correct message type
from sensor_msgs.msg import LaserScan
from go2_dyn_tube_mpc_msg.msg import Trajectory


class FakeSLAMNode(Node):
    def __init__(self):
        super().__init__('fake_slam')

        # Parameters
        self.declare_parameter('map_size', [200, 200])  # Grid size
        self.declare_parameter('resolution', 0.05)  # Map resolution
        self.declare_parameter('scan_range', 2.0)  # LIDAR range
        self.declare_parameter('scan_resolution', 0.1)  # Scan angle step
        self.declare_parameter('drift_rate', 0.001)  # Drift per second

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.map_update_pub = self.create_publisher(OccupancyGridUpdate, '/map_updates', 1)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 3)
        self.goal_pub = self.create_publisher(Pose2D, '/obelisk/go2/goal_pose', 1)

        self.mpc_sub = self.create_subscription(Trajectory, '/obelisk/go2/dtmpc_path', self.trajectory_callback, 1)

        # TF Broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Simulated Map
        self.map = self.generate_fake_map()

        # Ground Truth Trajectory
        self.trajectory = []
        self.current_index = 0

        self.z = np.zeros((3,))

        # TF Offsets
        self.map_to_odom_x = 0.0
        self.map_to_odom_y = 0.0
        self.map_to_odom_theta = 2.0

        # Timers
        self.create_timer(1.0, self.publish_map)
        self.create_timer(1.0, self.publish_goal_pose)
        self.create_timer(0.1, self.publish_scan)
        self.create_timer(0.1, self.publish_tf)

    def generate_fake_map(self):
        """Generate a simple occupancy grid."""
        size_x, size_y = self.get_parameter('map_size').value
        resolution = self.get_parameter('resolution').value
        data = np.zeros((size_y, size_x))
        data[:int(size_y // 5)] = -1
        data[size_y - int(size_y)// 10:] = 100

        y = size_y // 2 + 25
        x = size_x // 2 + 25
        data[y-5:y+5, x-5:x+5] = 100

        map_msg = OccupancyGrid()
        map_msg.header.frame_id = "map"
        # map_msg.info = MapMetaData()
        map_msg.info.width = size_x
        map_msg.info.height = size_y
        map_msg.info.resolution = resolution
        map_msg.info.origin.position.x = -size_x * resolution / 2
        map_msg.info.origin.position.y = -size_y * resolution / 2
        map_msg.data = data.flatten().astype(int).tolist()

        return map_msg

    def publish_map(self):
        """Publish the full map periodically."""
        self.map.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map)

        # Simulate small updates in a sub-region
        update = OccupancyGridUpdate()
        update.header.stamp = self.get_clock().now().to_msg()
        update.header.frame_id = "map"
        update.width = 10  # Small update region
        update.height = 10
        update.x = np.random.randint(0, self.map.info.width - update.width)
        update.y = np.random.randint(0, self.map.info.height - update.height)
        
        # Generate a random update in this region
        update.data = np.random.choice([0, 100, -1], update.width * update.height, p=[0.7, 0.2, 0.1]).tolist()
        
        # self.map_update_pub.publish(update)

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

    def publish_tf(self):
        """Publish TF transforms with drift in map->odom."""
        now = self.get_clock().now().to_msg()

        # Introduce Drift in map -> odom
        drift_rate = self.get_parameter('drift_rate').value
        self.map_to_odom_x += drift_rate * 0.25
        self.map_to_odom_y += drift_rate * 0.125
        self.map_to_odom_theta += drift_rate * .3

        tf_map_odom = TransformStamped()
        tf_map_odom.header.stamp = now
        tf_map_odom.header.frame_id = "map"
        tf_map_odom.child_frame_id = "odom"
        tf_map_odom.transform.translation.x = self.map_to_odom_x
        tf_map_odom.transform.translation.y = self.map_to_odom_y
        tf_map_odom.transform.rotation.z = math.sin(self.map_to_odom_theta / 2)
        tf_map_odom.transform.rotation.w = math.cos(self.map_to_odom_theta / 2)

        # Publish odom->base_link (True pose)
        if self.trajectory:
            pose_x, pose_y, theta = self.trajectory[self.current_index]
            tf_odom_base = TransformStamped()
            tf_odom_base.header.stamp = now
            tf_odom_base.header.frame_id = "odom"
            tf_odom_base.child_frame_id = "base_link"
            # tf_odom_base.transform.translation.x = pose_x
            # tf_odom_base.transform.translation.y = pose_y
            # tf_odom_base.transform.rotation.z = math.sin(theta / 2)
            # tf_odom_base.transform.rotation.w = math.cos(theta / 2)
            tf_odom_base.transform.translation.x = self.z[0]
            tf_odom_base.transform.translation.y = self.z[1]
            tf_odom_base.transform.rotation.z = math.sin(self.z[2] / 2)
            tf_odom_base.transform.rotation.w = math.cos(self.z[2] / 2)

            self.tf_broadcaster.sendTransform([tf_map_odom, tf_odom_base])
        else:
            self.tf_broadcaster.sendTransform([tf_map_odom])

    def publish_goal_pose(self):
        msg = Pose2D()
        msg.x = 2.
        msg.y = 2.
        msg.theta = np.pi / 4
        self.goal_pub.publish(msg)

    def set_trajectory(self, trajectory):
        """Allow user to set a trajectory [(x, y, theta), ...]."""
        self.trajectory = trajectory
        self.current_index = 0
        self.get_logger().info("Trajectory updated.")

def main(args=None):
    rclpy.init(args=args)
    node = FakeSLAMNode()

    # Example trajectory: Circular motion
    w = 10
    trajectory = [(math.cos(t) * 2, math.sin(t) * 2, t) for t in np.linspace(0, 2 * math.pi, 100 * w)]
    node.set_trajectory(trajectory)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
