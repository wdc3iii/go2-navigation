import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import LaserScan
from grid_map_msgs.msg import GridMap
from go2_dyn_tube_mpc_msgs.msg import Trajectory
from obelisk_control_msgs.msg import VelocityCommand
from geometry_msgs.msg import Point, TransformStamped
from obelisk_estimator_msgs.msg import EstimatedState
from visualization_msgs.msg import Marker, MarkerArray

from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound

from go2_dyn_tube_mpc.map_utils import map_to_pose
from go2_dyn_tube_mpc.dynamic_tube_mpc import DynamicTubeMPC

import time
import torch
import numpy as np
from scipy.io import savemat
from typing import List, Optional


class DynamicTubeMPCNode(ObeliskController):
    """Dynamic Tube MPC.
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
        self.n = 3
        self.m = 3

        self.path = None
        self.received_path = False
        self.received_map = False
        self.update_entire_map = True
        self.dtmpc_needs_warm_start_reset = True
        self.laser_to_odom_transform = self.get_identity_transform("odom", "base_link")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load policy
        # self.declare_parameter("tube_path", "")
        # tube_path = self.get_parameter("tube_path").get_parameter_value().string_value
        # self.tube_dyn = torch.load(tube_path)
        # self.device = next(self.tube_policy.parameters()).device
        # self.get_logger().info(f"Tube Dynamics: {tube_path} loaded on {self.device}. {len(self.kps)}, {len(self.kds)}")

        # Load DTMPC parameters
        # Horizon length, timing
        self.declare_parameter("N", 20)
        self.N = self.get_parameter("N").get_parameter_value().integer_value
        self.declare_parameter("robot_radius", 0.15)
        self.robot_radius = self.get_parameter("robot_radius").get_parameter_value().double_value
        self.N = self.get_parameter("N").get_parameter_value().integer_value
        self.declare_parameter("dt", 0.1)
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.ts = np.arange(0, self.N + 1) * self.dt

        # State variables
        self.px_z = np.zeros((3,))
        self.px_v = np.zeros((3,))

        # Velocity bounds
        self.declare_parameter("v_max", [1.0, 0.5, 1.0])
        self.v_max_bound = np.array(self.get_parameter("v_max").get_parameter_value().double_array_value)
        self.declare_parameter("v_min", [-0.1, -0.5, -1.0])
        self.v_min_bound = np.array(self.get_parameter("v_min").get_parameter_value().double_array_value)
        assert np.all(self.v_max_bound > 0) and np.all(self.v_min_bound <= 0)
        self.v_max = self.v_max_bound
        self.v_min = self.v_min_bound

        self.cost_map_msg = MarkerArray()

        # Cost matrices
        self.declare_parameter("Q", [1., 1., 0.5])          # State cost
        Q = np.array(self.get_parameter("Q").get_parameter_value().double_array_value)
        self.declare_parameter("Qf", [10., 10., 10.])       # Terminal Cost
        Qf = np.array(self.get_parameter("Qf").get_parameter_value().double_array_value)
        self.declare_parameter("R", [0.1, 0.1, 0.1])        # Temporal weights on state cost
        R = np.array(self.get_parameter("R").get_parameter_value().double_array_value)
        self.declare_parameter("Q_schedule", np.linspace(0, 1, self.N).tolist())    # Schedule on state cost (allow more deviation of trajectory from plan near beginning)
        Q_sched = np.array(self.get_parameter("Q_schedule").get_parameter_value().double_array_value)
        self.declare_parameter("Rv_first", [0.1, 0.1, 0.1])      # First order rate penalty on input
        Rv_first = np.array(self.get_parameter("Rv_first").get_parameter_value().double_array_value)
        self.declare_parameter("Rv_second", [0., 0., 0., 0., 0., 0., 0., 0., 0.])     # Second order rate penalty on input
        Rv_second = np.array(self.get_parameter("Rv_second").get_parameter_value().double_array_value)

        self.dtmpc = DynamicTubeMPC(self.dt, self.N, self.n, self.m, Q, Qf, R, Rv_first, Rv_second, Q_sched, self.v_min_bound, self.v_max_bound, robot_radius=self.robot_radius)

        # Declare subscriber to high level plans
        self.register_obk_subscription(
            "sub_plan_setting",
            self.plan_callback,  # type: ignore
            key="sub_plan_key",  # key can be specified here or in the config file
            msg_type=Trajectory
        )
        # Subscriber to velocity limits from high level
        self.register_obk_subscription(
            "sub_vel_lim_setting",
            self.vel_lim_callback,  # type: ignore
            key="sub_vel_lim_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )
        # Subscriber to map update
        self.register_obk_subscription(
            "sub_nearest_points_setting",
            self.nearest_points_callback,  # type: ignore
            key="sub_nearest_points_key",  # key can be specified here or in the config file
            msg_type=GridMap
        )
        # Subscriber to laser scan
        self.register_obk_subscription(
            "sub_scan_setting",
            self.laser_scan_callback,  # type: ignore
            key="sub_scan_key",  # key can be specified here or in the config file
            msg_type=LaserScan
        )
        self.register_obk_publisher(
            "pub_viz_setting",
            key="pub_viz_key",  # key can be specified here or in the config file
            msg_type=Marker
        )
        self.register_obk_publisher(
            "pub_constraint_setting",
            key="pub_constraint_key",  # key can be specified here or in the config file
            msg_type=Marker
        )

        self.last_warn = self.get_clock().now()
        self.get_logger().info("DTMPC node constructor complete.")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)        
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """

        # if len(x_hat_msg.q_base) == 7:
        #     self.px_z = np.array([
        #         x_hat_msg.q_base[0],                    # x
        #         x_hat_msg.q_base[1],                    # y
        #         self.quat2yaw(x_hat_msg.q_base[3:])     # theta
        #     ])
        # else:
        #     self.get_logger().error(f"Estimated State base pose size does not match URDF! Size is {len(x_hat_msg.q_base)} instead of 7.")

        # if len(x_hat_msg.v_base) == 6:
        #     self.px_v = np.array([
        #         x_hat_msg.v_base[0],                    # v_x
        #         x_hat_msg.v_base[1],                    # v_y
        #         x_hat_msg.v_base[-1]                    # w_z
        #     ])
        # else:
        #     self.get_logger().error(f"Estimated State base velocity size does not match URDF! Size is {len(x_hat_msg.v_base)} instead of 6.")

        # self.dtmpc.set_initial_condition(self.px_z)
        """Going to use tf listener for now"""
        pass

    def plan_callback(self, plan_msg: Trajectory):
        # Accepts a trajectory which is a sequence of waypoints
        self.dtmpc.set_path(np.array(plan_msg.z).reshape(plan_msg.horizon + 1, plan_msg.n))
        self.received_path = True

        if self.dtmpc_needs_warm_start_reset:
            self.dtmpc.reset_warm_start()
            self.dtmpc_needs_warm_start_reset = False

    def nearest_points_callback(self, nearest_points_msg: GridMap):
        dim0 = nearest_points_msg.data[0].layout.dim
        nearest_dists = np.array(nearest_points_msg.data[0].data).reshape(dim0[0].size, dim0[1].size)
        
        dim1 = nearest_points_msg.data[1].layout.dim
        dim2 = nearest_points_msg.data[2].layout.dim
        nearest_inds = np.concatenate((
            np.array(nearest_points_msg.data[1].data).reshape(1, dim1[0].size, dim1[1].size).astype(int),
            np.array(nearest_points_msg.data[2].data).reshape(1, dim2[0].size, dim2[1].size).astype(int)
        ), axis=0)

        map_origin = np.array([nearest_points_msg.info.pose.position.x, nearest_points_msg.info.pose.position.y])
        map_theta = self.quat2yaw((
            nearest_points_msg.info.pose.orientation.x,
            nearest_points_msg.info.pose.orientation.y,
            nearest_points_msg.info.pose.orientation.z,
            nearest_points_msg.info.pose.orientation.w
        ))
        self.dtmpc.update_nearest_inds(nearest_inds, nearest_dists, map_origin, map_theta)
        self.received_map = True

        savemat("nearest_data.mat", {"nearest_dists": nearest_dists, "nearest_inds": nearest_inds, "map_origin": map_origin, "map_theta": map_theta})
        # raise RuntimeError()

    def laser_scan_callback(self, scan_msg: LaserScan):
        # Convert the laser scan into the odom frame
        try:
            self.laser_to_odom_transform = self.tf_buffer.lookup_transform("odom", scan_msg.header.frame_id, rclpy.time.Time(), Duration(seconds=1.0))
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Transform error: {e}: odom -> {scan_msg.header.frame_id}")
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)

        # Convert to Cartesian (laser frame)
        valid = np.isfinite(ranges)  # Ignore NaN and inf values
        x_laser = ranges[valid] * np.cos(angles[valid])
        y_laser = ranges[valid] * np.sin(angles[valid])
        points_laser = np.vstack((x_laser, y_laser, np.ones_like(x_laser)))  # Homogeneous coordinates

        # Extract transformation matrix
        H = self.transform_to_matrix(self.laser_to_odom_transform)

        # Transform points to odom frame
        scan_points = np.dot(H, points_laser)[:2].T  # Extract x, y
        self.dtmpc.update_scan(scan_points)
    
    def transform_to_matrix(self, transform):
        """Convert a ROS2 TransformStamped into a 3x3 homogeneous transformation matrix (2D)."""
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to yaw angle
        yaw = self.quat2yaw([rotation.x, rotation.y, rotation.z, rotation.w])

        # Construct 2D transformation matrix
        H = np.array([
            [np.cos(yaw), -np.sin(yaw), translation.x],
            [np.sin(yaw),  np.cos(yaw), translation.y],
            [0, 0, 1]
        ])
        
        return H

    def vel_lim_callback(self, v_max_msg: VelocityCommand):
        """Sets the upper bound on velocity limits for the planner"""
        prop_v_lim = np.clip(np.array([v_max_msg.v_x, v_max_msg.v_y, v_max_msg.w_z]), 0., 1.)
        self.v_max = prop_v_lim * self.v_max_bound
        self.v_min = prop_v_lim * self.v_min_bound

        self.dtmpc.set_input_bounds(self.v_min, self.v_max)

    @staticmethod
    def quat2yaw(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def compute_control(self) -> Trajectory:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Don't plan if we haven't received a path to follow
        traj_msg = Trajectory()
        msg_time = self.get_clock().now()
        traj_msg.header.stamp = msg_time.to_msg()
        if not (self.received_path and self.received_map):
            if (msg_time - self.last_warn).nanoseconds > 1e9:
                self.get_logger().warn(f"Have not recieved path {self.received_path} or map {self.received_map}. Cannot run DTMPC")
                self.last_warn = msg_time
            return traj_msg

        # Solve the MPC
        try:
            odom_to_base = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time(), Duration(seconds=1.0))
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Transform error: {e}: odom -> base_link")
            return 

        yaw = self.quat2yaw([odom_to_base.transform.rotation.x, odom_to_base.transform.rotation.y, odom_to_base.transform.rotation.z, odom_to_base.transform.rotation.w])
        self.dtmpc.set_initial_condition(np.array([odom_to_base.transform.translation.x, odom_to_base.transform.translation.y, yaw]))
        t0 = time.perf_counter_ns()
        z, v, info = self.dtmpc.solve()
        self.get_logger().info(f"DTMPC solve took {(time.perf_counter_ns() - t0) * 1e-9}: Status {info}")

        # Construct the message
        traj_msg.horizon = self.dtmpc.N
        traj_msg.n = self.dtmpc.n
        traj_msg.m = self.dtmpc.m
        traj_msg.z = z.flatten().tolist()
        traj_msg.v = v.flatten().tolist()
        traj_msg.t = self.ts.tolist()
        
        self.obk_publishers["pub_ctrl"].publish(traj_msg)

        A, b, nearest_points = self.dtmpc.compute_constraints()
        constraint_msg = Marker()
        constraint_msg.header.stamp = msg_time.to_msg()
        constraint_msg.header.frame_id = 'odom'
        constraint_msg.ns = "dynamic_tube_mpc"
        constraint_msg.type = Marker.LINE_LIST
        constraint_msg.action = Marker.ADD
        constraint_msg.scale.x = 0.02
        constraint_msg.scale.y = 0.02
        constraint_msg.scale.z = 0.02
        clr = ColorRGBA()
        clr.a = 0.5
        clr.r = 0.
        clr.b = 1.
        clr.g = 0.
        for i in range(b.shape[0]):
            a0 = A[i, :]
            b0 = b[i]

            p = nearest_points[i, :]
            c = 1
            p1 = Point()
            p2 = Point()
            if np.abs(a0[0]) > 0.5:
                p1.y = p[1] - c
                p2.y = p[1] + c
                p1.x = -(a0[1] * p1.y + b0) / a0[0]
                p2.x = -(a0[1] * p2.y + b0) / a0[0]
            else:
                p1.x = p[0] - c
                p2.x = p[0] + c
                p1.y = -(a0[0] * p1.x + b0) / a0[1]
                p2.y = -(a0[0] * p2.x + b0) / a0[1]
            
            constraint_msg.points.append(p1)
            constraint_msg.points.append(p2)
            constraint_msg.colors.append(clr)
            constraint_msg.colors.append(clr)
        self.obk_publishers["pub_constraint_key"].publish(constraint_msg)

        # Publish message for viz
        viz_msg = Marker()
        viz_msg.header.stamp = msg_time.to_msg()
        viz_msg.header.frame_id = 'odom'
        viz_msg.ns = "dynamic_tube_mpc"
        viz_msg.type = Marker.LINE_STRIP
        viz_msg.action = Marker.ADD
        viz_msg.scale.x = 0.02
        viz_msg.scale.y = 0.02
        viz_msg.scale.z = 0.02
        for i in range(z.shape[0]):
            point = Point()
            point.x = z[i, 0]
            point.y = z[i, 1]
            viz_msg.points.append(point)
            color = ColorRGBA()
            color.a = 1.
            color.r = 0.
            color.b = (z.shape[0] - 1 - i) / (z.shape[0] - 1)
            color.g = i / (z.shape[0] - 1)
            viz_msg.colors.append(color)

        self.obk_publishers["pub_viz_key"].publish(viz_msg)


        # for marker in self.cost_map_msg.markers:
        #     marker.action = Marker.DELETE
        # self.obk_publishers['pub_dist_key'].publish(self.cost_map_msg)
        # self.cost_map_msg = MarkerArray()
        # i = 0
        # for xc in range(self.dtmpc.map_nearest_dists.shape[0]):
        #     for yc in range(self.dtmpc.map_nearest_dists.shape[1]):
        #         pos = map_to_pose(np.array((xc, yc)), self.dtmpc.map_origin, self.dtmpc.map_theta, self.dtmpc.map_resolution)
        #         dist = Marker()
        #         dist.header.stamp = msg_time.to_msg()
        #         dist.header.frame_id = 'map'
        #         dist.type = Marker.CUBE
        #         dist.ns = "dist map"
        #         dist.id = i
        #         i += 1
        #         dist.action = Marker.ADD
        #         dist.scale.x = 0.05
        #         dist.scale.y = 0.05
        #         dist.scale.z = 0.01
        #         dist.color.a = 0.5
        #         d = self.dtmpc.map_nearest_dists[xc, yc].astype(float)
                
        #         frac = min(abs(d), 1.)
        #         if d > 0:
        #             dist.color.r = 0.
        #             dist.color.b = 1. - frac
        #             dist.color.g = frac
        #         else:
        #             dist.color.r = frac
        #             dist.color.b = 1. - frac
        #             dist.color.g = 0.
        #         dist.pose.position.x = pos[0]
        #         dist.pose.position.y = pos[1]
        #         self.cost_map_msg.markers.append(dist)
        # self.obk_publishers['pub_dist_key'].publish(self.cost_map_msg)
        # assert is_in_bound(type(traj_msg), ObeliskControlMsg)
        return traj_msg

    def get_identity_transform(self, parent_frame, child_frame):
        identity_transform = TransformStamped()
        identity_transform.header.stamp = self.get_clock().now().to_msg()
        identity_transform.header.frame_id = parent_frame
        identity_transform.child_frame_id = child_frame

        # No translation (x, y, z = 0)
        identity_transform.transform.translation.x = 0.0
        identity_transform.transform.translation.y = 0.0
        identity_transform.transform.translation.z = 0.0

        # Identity quaternion (no rotation)
        identity_transform.transform.rotation.x = 0.0
        identity_transform.transform.rotation.y = 0.0
        identity_transform.transform.rotation.z = 0.0
        identity_transform.transform.rotation.w = 1.0

        return identity_transform


def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, DynamicTubeMPCNode, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
