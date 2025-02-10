from typing import List, Optional

import tf2_ros
from rclpy.executors import MultiThreadedExecutor
import rclpy
from rclpy.duration import Duration
from obelisk_py.core.utils.ros import spin_obelisk
import numpy as np
from go2_dyn_tube_mpc_msg.msg import Trajectory
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_control_msgs.msg import VelocityCommand
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound

from go2_dyn_tube_mpc.go2_dyn_tube_mpc.high_level_planner import HighLevelPlanner
from geometry_msgs.msg import Pose2D, Point
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import time
import threading 


class HighLevelPlannerNode(ObeliskController):
    """High Level Planner (accomplishes goal finding and frontier exploration).
    pub: publishes optimized trajectory
    sub: estimated state of robot
    sub: high level plan "sub_plan_setting"
    sub: proportion of velocity limits "sub_vel_lim_setting"
    sub: map "sub_map_setting"
    sub: scan "sub_scan_setting"
    """

    def __init__(self, node_name: str = "dynamic_tube_mpc_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, Trajectory, EstimatedState)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create high level planner
        self.declare_parameter("min_frontier_size", 5)
        self.map_lock = threading.Lock()
        self.inflated_map_lock = threading.Lock()
        self.explorer = HighLevelPlanner(
            self.map_lock, self.inflated_map_lock,
            min_frontier_size=self.get_parameter("min_frontier_size").get_parameter_value().integer_value,
            free=0, uncertain=-1, occupied=100
        )
        # self.declare_parameter("downsample", 4)
        # self.downsample = self.get_parameter("downsample").get_parameter_value().integer_value

        # proportion of maximum planning velocity to use while mapping
        self.declare_parameter("v_max_mapping", 0.2)
        self.v_max_mapping = self.get_parameter("v_max_mapping").get_parameter_value().double_value

        # Nearest_points_size
        self.declare_parameter("nearest_points_size", 100)
        self.nearest_points_size = self.get_parameter("nearest_points_size").get_parameter_value().integer_value

        # Local variables
        self.goal_pose = np.zeros((3,))     # Goal to navigate to
        self.px_z = np.zeros((3,))          # Current pose of the robot (in the odom frame)
        self.received_goal = False          # Whether a goal has been recieved
        self.received_map = False           # Whether a map has been recieved
        self.update_entire_map = True       # When an entire map update needs to occur
        self.map_to_odom_p = None           # position from map -> odom transform
        self.map_to_odom_yaw = None         # yaw 

        self.front_msg = MarkerArray()

        # Declare subscriber to pose commands
        self.register_obk_subscription(
            "sub_goal_setting",
            self.goal_callback,  # type: ignore
            key="sub_goal_key",  # key can be specified here or in the config file
            msg_type=Pose2D
        )
        self.register_obk_subscription(
            "sub_map_update_setting",
            self.map_update_callback,  # type: ignore
            key="sub_map_update_key",  # key can be specified here or in the config file
            msg_type=OccupancyGridUpdate
        )
        self.register_obk_subscription(
            "sub_map_setting",
            self.map_callback,  # type: ignore
            key="sub_map_key",  # key can be specified here or in the config file
            msg_type=OccupancyGrid
        )
        self.register_obk_publisher(
            "pub_vel_lim_setting",
            key="pub_vel_lim_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )
        self.register_obk_publisher(
            "pub_nearest_points_setting",
            key="pub_nearest_points_key",  # key can be specified here or in the config file
            msg_type=GridMap
        )
        self.register_obk_publisher(
            "pub_viz_setting",
            key="pub_viz_key",  # key can be specified here or in the config file
            msg_type=Marker
        )
        self.register_obk_publisher(
            "pub_front_setting",
            key="pub_front_key",  # key can be specified here or in the config file
            msg_type=MarkerArray
        )
        self.register_obk_timer(
            "timer_nearest_pts_setting",
            self.pub_nearest_points_callback,
            key="timer_nearest_pts_key"
        )

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)

        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """

        pass

    def goal_callback(self, goal_msg: Pose2D):
        # Accepts a trajectory which is a sequence of waypoints
        self.goal_pose = np.array([goal_msg.x, goal_msg.y, goal_msg.theta])
        self.received_goal = True

    def map_callback(self, map_msg: OccupancyGrid):
        if self.update_entire_map:
            occ_grid = np.array(map_msg.data, dtype=np.int8).reshape(map_msg.info.height, map_msg.info.width)
            th = self.quat2yaw([map_msg.info.origin.orientation.x, map_msg.info.origin.orientation.y, map_msg.info.origin.orientation.z, map_msg.info.origin.orientation.w])
            origin = np.array([map_msg.info.origin.position.x, map_msg.info.origin.position.y, th])
            self.explorer.set_map(occ_grid, origin, map_msg.info.resolution)
            self.received_map = True
            self.update_entire_map = False

    def map_update_callback(self, map_update_msg: OccupancyGridUpdate):
        x = map_update_msg.x
        y = map_update_msg.y
        h = map_update_msg.height
        w = map_update_msg.width
        if self.update_entire_map or x < 0 or y < 0 or x + w >= self.explorer.map.shape[0] or y + h >= self.explorer.map.shape[1]:
            self.update_entire_map = True
            return
        update_occ = np.array(map_update_msg.data, dtype=np.int8).reshape(h, w)
        # self.explorer.update_map(update_occ, (x, y))
        self.explorer.update_map(update_occ, (x, y))

    def pub_nearest_points_callback(self):
        if not self.received_map:
            self.get_logger().warn("High Level has not received map yet, and cannot compute nearest points.")
            return
        
        try:
            # Get transform from odom to robot frame
            map_to_odom = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time(), Duration(seconds=10.0))

            # Extract robot pose in odom
            self.map_to_odom_p = np.array([
                map_to_odom.transform.translation.x, map_to_odom.transform.translation.y
            ])
            self.map_to_odom_yaw = self.quat2yaw([map_to_odom.transform.rotation.x, map_to_odom.transform.rotation.y, map_to_odom.transform.rotation.z, map_to_odom.transform.rotation.w])

            map_to_base = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), Duration(seconds=1.0))
        except tf2_ros.LookupException as e:
            self.get_logger().warm(f"Transform error: {e}")
            return

        # Compute submaps
        pz_x_map = np.array([map_to_base.transform.translation.x, map_to_base.transform.translation.y])
        nearest_inds, nearest_dists, sub_map_origin, sub_map_yaw = self.explorer.compute_nearest_inds(pz_x_map, self.nearest_points_size)

        nearest_point_msg = GridMap()
        # Header
        nearest_point_msg.header.stamp = self.get_clock().now().to_msg()
        nearest_point_msg.header.frame_id = "odom"

        # Layers and info
        nearest_point_msg.layers = ['sdf', 'nearest_x', 'nearest_y']
        nearest_point_msg.info.resolution = self.explorer.map.resolution
        nearest_point_msg.info.length_x = nearest_dists.shape[0] * self.explorer.map.resolution
        nearest_point_msg.info.length_y = nearest_dists.shape[1] * self.explorer.map.resolution
        # Convert origin and yaw to odom frame
        sub_map_odom_origin = np.array([
            [np.cos(sub_map_yaw - self.map_to_odom_yaw), np.sin(sub_map_yaw - self.map_to_odom_yaw)],
            [-np.sin(sub_map_yaw - self.map_to_odom_yaw), np.cos(sub_map_yaw - self.map_to_odom_yaw)],
        ]) @ (sub_map_origin - self.map_to_odom_p)

        nearest_point_msg.info.pose.position.x = sub_map_odom_origin[0]
        nearest_point_msg.info.pose.position.y = sub_map_odom_origin[1]
        quat = self.yaw2quat(sub_map_yaw - self.map_to_odom_yaw)
        nearest_point_msg.info.pose.orientation.x = quat[0]
        nearest_point_msg.info.pose.orientation.y = quat[1]
        nearest_point_msg.info.pose.orientation.z = quat[2]
        nearest_point_msg.info.pose.orientation.w = quat[3]

        # Depth (Z dimension)
        depth_dim = MultiArrayDimension()
        depth_dim.label = "layers"
        depth_dim.size = 3
        depth_dim.stride = 3 * nearest_dists.shape[0] * nearest_dists.shape[1]  # Total elements
        # Height (Y dimension)
        height_dim = MultiArrayDimension()
        height_dim.label = "height"
        height_dim.size = nearest_dists.shape[0]
        height_dim.stride = nearest_dists.shape[0] * nearest_dists.shape[1]  # Elements per depth slice
        # Width (X dimension)
        width_dim = MultiArrayDimension()
        width_dim.label = "width"
        width_dim.size = nearest_dists.shape[1]
        width_dim.stride = nearest_dists.shape[1]  # Elements per row

        # Data
        nearest_point_msg.data = []
        stacked_data = np.concatenate((nearest_dists[None, :, :], nearest_inds), axis=0)
        for i in range(stacked_data.shape[0]):  # Iterate over depth (3 layers)
            multi_array = Float32MultiArray()
            multi_array.layout = MultiArrayLayout()
            
            # Define layout (2D grid)
            dim_x = MultiArrayDimension(label="height", size=stacked_data.shape[1], stride=stacked_data.shape[1] * stacked_data.shape[2])
            dim_y = MultiArrayDimension(label="width", size=stacked_data.shape[2], stride=stacked_data.shape[2])

            multi_array.layout.dim = [dim_x, dim_y]
            multi_array.layout.data_offset = 0
            
            # Convert the 2D array to a flattened list
            multi_array.data = stacked_data[i].flatten().tolist()
            
            nearest_point_msg.data.append(multi_array)
        self.obk_publishers["pub_nearest_points_key"].publish(nearest_point_msg)

    @staticmethod
    def quat2yaw(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    @staticmethod
    def yaw2quat(yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return [0., 0., sy, cy]  # (x, y, z, w)
    
    def compute_control(self) -> Trajectory:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Don't plan if we haven't received a path to follow
        traj_msg = Trajectory()
        msg_time = self.get_clock().now()
        traj_msg.header.stamp = msg_time.to_msg()
        if not self.received_goal:
            self.get_logger().warn("High Level Planner has not recieved goal")
            return traj_msg
        try:
            map_to_base = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), Duration(seconds=1.0))
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Transform error: {e}: map -> base_link")
            return traj_msg
        
        t0 = time.perf_counter_ns()
        path, cost, frontiers, info = self.explorer.find_frontiers_to_goal(np.array([map_to_base.transform.translation.x, map_to_base.transform.translation.y, 0]), self.goal_pose, self.downsample)
        self.get_logger().info(f"Astar solve took {(time.perf_counter_ns() - t0) * 1e-9}\tResult: {info}")

        if info == self.explorer.PATH_START_NOT_FREE:
            self.get_logger().warn("High Level Plan began in occupied space!")
            return traj_msg
        elif info == self.explorer.NO_PLAN_FOUND:
            self.get_logger().warn("High Level Plan not found - no frontiers or goal!")
            return traj_msg
        
        # TODO: decide whether to follow path in Dynamic or Mapping mode
        vel_lim = VelocityCommand()
        vel_lim.v_x = 1.
        vel_lim.v_y = 1.
        vel_lim.w_z = 1.
        self.obk_publishers["pub_vel_lim_key"].publish(vel_lim)

        try:
            # Get transform from odom to robot frame
            map_to_odom = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time(), Duration(seconds=1.0))

            # Extract robot pose in odom
            self.map_to_odom_p = np.array([
                map_to_odom.transform.translation.x, map_to_odom.transform.translation.y
            ])
            self.map_to_odom_yaw = self.quat2yaw([map_to_odom.transform.rotation.x, map_to_odom.transform.rotation.y, map_to_odom.transform.rotation.z, map_to_odom.transform.rotation.w])
        except tf2_ros.LookupException as e:
            self.get_logger().warm(f"Transform error: {e}")
            return
        path = (np.array([
            [np.cos(self.map_to_odom_yaw), np.sin(self.map_to_odom_yaw)],
            [-np.sin(self.map_to_odom_yaw), np.cos(self.map_to_odom_yaw)],
        ]) @ (path - self.map_to_odom_p).T).T
        
        # setting the message (geometric only path)
        traj_msg.horizon = path.shape[0] - 1
        traj_msg.n = path.shape[1]
        traj_msg.m = 0
        traj_msg.z = np.array(path).flatten().tolist()

        self.obk_publishers["pub_ctrl"].publish(traj_msg)
        
        # Visualize frontiers
        yel = [0.9290, 0.6940, 0.1250]
        pur = [0.4940, 0.1840, 0.5560]
        
        for marker in self.front_msg.markers:
            marker.action = Marker.DELETE
        self.obk_publishers["pub_front_key"].publish(self.front_msg)
        self.front_msg = MarkerArray()
        if frontiers:
            for ind, front in enumerate(frontiers):
                front_map = self.explorer.map.map_to_pose(front[0])
                frac = ind / (max(len(front) - 1, 1))
                r = frac * pur[0] + (1 - frac) * yel[0]
                g = frac * pur[1] + (1 - frac) * yel[1]
                b = frac * pur[2] + (1 - frac) * yel[2]
                for pt_ind, front_pt in enumerate(front_map):
                    pt = Marker()
                    pt.header.stamp = msg_time.to_msg()
                    pt.header.frame_id = 'map'
                    pt.type = Marker.SPHERE
                    pt.ns = f"frontier_{ind}"
                    pt.id = pt_ind
                    pt.action = Marker.ADD
                    pt.scale.x = 0.05
                    pt.scale.y = 0.05
                    pt.scale.z = 0.05
                    pt.color.a = 1.
                    pt.color.r = r
                    pt.color.b = b
                    pt.color.g = g
                    pt.pose.position.x = front_pt[0]
                    pt.pose.position.y = front_pt[1]
                    self.front_msg.markers.append(pt)
            self.obk_publishers["pub_front_key"].publish(self.front_msg)
        else:
            self.get_logger().info("No frontiers located.")

        # Publish message for viz
        viz_msg = Marker()
        viz_msg.header.stamp = msg_time.to_msg()
        viz_msg.header.frame_id = 'odom'
        viz_msg.type = Marker.LINE_STRIP
        viz_msg.ns = "high_level_planning"
        viz_msg.action = Marker.ADD
        viz_msg.scale.x = 0.02
        viz_msg.scale.y = 0.02
        viz_msg.scale.z = 0.02
        for i in range(path.shape[0]):
            point = Point()
            point.x = path[i, 0]
            point.y = path[i, 1]
            viz_msg.points.append(point)
            color = ColorRGBA()
            color.a = 1.
            color.r = 1.
            color.b = 0.
            color.g = 0.
            viz_msg.colors.append(color)

        self.obk_publishers["pub_viz_key"].publish(viz_msg)
        # assert is_in_bound(type(traj_msg), ObeliskControlMsg)
        return traj_msg


def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, HighLevelPlannerNode, MultiThreadedExecutor)


if __name__ == "__main__":
    main()

