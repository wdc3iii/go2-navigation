import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from go2_dyn_tube_mpc_msgs.msg import Trajectory
from obelisk_control_msgs.msg import VelocityCommand

from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound

from go2_dyn_tube_mpc.trajectory_tracker import track_trajectory

import numpy as np
from typing import List, Optional


class TrajectoryTracker(ObeliskController):
    """Trajectory Tracker.
    pub: publishes planar velocity command to send to the robot
    sub: trajectory to track
    """

    def __init__(self, node_name: str = "trajectory_tracker") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, VelocityCommand, Trajectory)

        # Storing path
        self.z_des = None
        self.v_des = None
        self.t_path = None
        self.t_last_path = False

        # Listen to TF for robot position in the odom frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Feedback gains
        self.declare_parameter("kp", [1., 1., 1.])
        self.K = np.array(self.get_parameter("kp").get_parameter_value().double_array_value)

        # Velocity bounds
        self.declare_parameter("v_max", [1.0, 1.0, 1.0])
        self.v_max = np.array(self.get_parameter("v_max").get_parameter_value().double_array_value)
        self.declare_parameter("v_min", [-1.0, -1.0, -1.0])
        self.v_min = np.array(self.get_parameter("v_min").get_parameter_value().double_array_value)
        assert np.all(self.v_max > 0) and np.all(self.v_min <= 0)

        self.last_warn = self.get_clock().now()
        self.get_logger().info("Tracjectory tracking node constructor complete.")


    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)
        
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, plan_msg: Trajectory):
        # Accepts a trajectory which is a sequence of waypoints
        self.z_des = np.array(plan_msg.z).reshape(plan_msg.horizon + 1, plan_msg.n)
        self.v_des = np.array(plan_msg.v).reshape(plan_msg.horizon, plan_msg.m)
        self.t_path = np.array(plan_msg.t)
        self.t_last_path = self.get_clock().now()
        

    def compute_control(self) -> Trajectory:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Don't plan if we haven't received a path to follow
        traj_msg = VelocityCommand()
        msg_time = self.get_clock().now()
        traj_msg.header.stamp = msg_time.to_msg()
        if self.t_path is None:
            if (msg_time - self.last_warn).nanoseconds > 1e9:
                self.get_logger().warn("Have not recieved path. Cannot run Tracking Controller.")
                self.last_warn = msg_time
            return traj_msg

        # Compute the velocity to follow path
        try:
            odom_to_base = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time(), Duration(seconds=1.0))
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"Transform error: {e}: odom -> base_link")
            return 

        pz_x = np.array((
            odom_to_base.transform.translation.x,
            odom_to_base.transform.translation.y,
            self.quat2yaw([odom_to_base.transform.rotation.x, odom_to_base.transform.rotation.y, odom_to_base.transform.rotation.z, odom_to_base.transform.rotation.w])
        ))
        t = (self.get_clock().now() - self.t_last_path).nanoseconds / 1e9

        v_cmd = track_trajectory(self.z_des, self.v_des, self.t_path, t, pz_x, self.K, self.v_min, self.v_max)

        # Construct the message
        traj_msg.v_x = v_cmd[0]
        traj_msg.v_y = v_cmd[1]
        traj_msg.w_z = v_cmd[2]
        
        self.obk_publishers["pub_ctrl"].publish(traj_msg)

        assert is_in_bound(type(traj_msg), ObeliskControlMsg)
        return traj_msg

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


def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, TrajectoryTracker, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
