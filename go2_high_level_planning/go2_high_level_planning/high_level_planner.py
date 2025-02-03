from typing import List, Optional

from rclpy.executors import SingleThreadedExecutor
from obelisk_py.core.utils.ros import spin_obelisk
import numpy as np
from go2_dyn_tube_mpc_msg.msg import Trajectory
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_control_msgs.msg import VelocityCommand
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound

from go2_high_level_planning.exploration import Exploration
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid, OccupancyGridUpdate


class HighLevelPlannerNode(ObeliskController):
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

        # Load DTMPC parameters
        # Horizon length, timing
        self.declare_parameter("min_frontier_size", 5)
        self.explorer = Exploration(
            min_frontier_size=self.get_parameter("min_frontier_size").integer_value,
            free=0, uncertain=1, occupied=100
        )

        # Velocity bounds
        self.declare_parameter("v_max_dyn", [0.2, 0.2, 0.2])
        self.v_max_dyn = np.array(self.get_parameter("v_max_dyn").get_parameter_value().double_array_value)
        self.declare_parameter("v_min_dyn", [-0.1, -0.2, -0.2])
        self.v_min_dyn = np.array(self.get_parameter("v_min_dyn").get_parameter_value().double_array_value)
        assert np.all(self.v_max_dyn > 0) and np.all(self.v_min_dyn <= 0)

        self.declare_parameter("v_max_map", [0.5, 0.5, 0.5])
        self.v_max_map = np.array(self.get_parameter("v_max_map").get_parameter_value().double_array_value)
        self.declare_parameter("v_min_map", [-0.1, -0.5, -0.5])
        self.v_min_map = np.array(self.get_parameter("v_min_map").get_parameter_value().double_array_value)
        assert np.all(self.v_max_map > 0) and np.all(self.v_min_map <= 0)

        self.goal_pose = np.zeros((3,))
        self.received_goal = False
        self.received_curr = False

        # TODO: get initial map
        # Declare subscriber to pose commands
        self.register_obk_subscription(
            "sub_goal_setting",
            self.goal_callback,  # type: ignore
            key="sub_goal_key",  # key can be specified here or in the config file
            msg_type=Pose2D
        )
        self.register_obk_subscription(
            "sub_map_setting",
            self.map_update_callback,  # type: ignore
            key="sub_map_key",  # key can be specified here or in the config file
            msg_type=OccupancyGridUpdate
        )
        self.register_obk_publisher(
            "pub_vel_lim_setting",
            key="pub_vel_lim_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
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

        if len(x_hat_msg.q_base) == 7:
            self.px_z = np.array([
                x_hat_msg.q_base[0],  # x
                x_hat_msg.q_base[1],  # y
                self.quat2yaw(x_hat_msg.q_base[3:])  # theta
            ])
            self.received_curr = True
        else:
            self.get_logger().error(
                f"Estimated State base pose size does not match URDF! Size is {len(x_hat_msg.q_base)} instead of 7.")

    def goal_callback(self, goal_msg: Pose2D):
        # Accepts a trajectory which is a sequence of waypoints
        self.goal_pose = np.array([goal_msg.x, goal_msg.y, goal_msg.theta])
        self.received_goal = True

    def map_update_callback(self, map_update_msg: OccupancyGridUpdate):
        # TODO: update the map of the explorer
        pass

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

    def compute_control(self) -> Trajectory:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Don't plan if we haven't received a path to follow
        if not self.received_goal or not self.received_curr:
            return
        path, cost, frontiers = self.explorer.find_frontiers_to_goal(self.curr_pose, self.goal_pose)

        # TODO: decide whether to follow path in Dynamic or Mapping mode

        # setting the message (geometric only path)
        # TODO: should add desired orientation, at least at goal...
        traj_msg = Trajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.z = path

        self.obk_publishers["pub_ctrl"].publish(traj_msg)
        assert is_in_bound(type(traj_msg), ObeliskControlMsg)
        return traj_msg


def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, HighLevelPlannerNode, SingleThreadedExecutor)


if __name__ == "__main__":
    main()

