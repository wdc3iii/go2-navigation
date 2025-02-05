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

from go2_dyn_tube_mpc.exploration import Exploration
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
        self.declare_parameter("v_max_mapping", 0.2)
        self.v_max_mapping = self.get_parameter("v_max_mapping").get_parameter_value().double_value

        self.goal_pose = np.zeros((3,))
        self.px_z = np.zeros((3,))
        self.received_goal = False
        self.received_curr = False
        self.received_map = False
        self.update_entire_map = True

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
            msg_type=OccupancyGridUpdate
        )
        self.register_obk_publisher(
            "pub_vel_lim_setting",
            key="pub_vel_lim_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )
        self.register_obk_publisher(
            "pub_nearest_points_setting",
            key="pub_nearest_points_key",  # key can be specified here or in the config file
            msg_type=OccupancyGrid
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

    def map_callback(self, map_msg: OccupancyGrid):
        if self.update_entire_map:
            occ_grid = np.array(map_msg.data, dtype=np.int8).reshape(map_msg.info.height, map_msg.info.width)
            th = self.quat2yaw([map_msg.orientation.x, map_msg.orientation.y, map_msg.orientation.z, map_msg.orientation.w])
            origin = np.array([map_msg.info.origin.position.x, map_msg.info.origin.position.y, th])
            self.explorer.set_map(occ_grid, origin, map_msg.info.resolution)
            self.received_map = True
            self.update_entire_map = False

    def map_update_callback(self, map_update_msg: OccupancyGridUpdate):
        x = map_update_msg.x
        y = map_update_msg.y
        h = map_update_msg.height
        w = map_update_msg.width
        if x < 0 or y < 0 or x + w >= self.map.shape[0] or y + h >= self.map.shape[1] or self.update_entire_map:
            self.update_entire_map = True
            return
        update_occ = np.array(map_update_msg.data, dtype=np.int8).reshape(h, w)
        self.explorer.update_map(update_occ, (x, y))

    def pub_nearest_points_callback(self):  # TODO: publish at 10Hz
        # TODO: Frames properly
        nearest_points = self.explorer.compute_nearest_inds(self.px_z[:2], self.nearest_points_size)

        nearest_point_msg = OccupancyGrid()
        nearest_point_msg.header.stamp = self.get_clock().now().to_msg()
        nearest_point_msg.header.frame_id = "map"
        nearest_point_msg.info.resolution = self.explorer.map.resolution
        nearest_point_msg.info.width = self.nearest_points_size
        nearest_point_msg.info.height = self.nearest_points_size
        nearest_point_msg.info.origin.position.x = ...
        nearest_point_msg.info.origin.position.y = ...
        nearest_point_msg.info.origin.orientation.x = ...
        nearest_point_msg.info.origin.orientation.y = ...
        nearest_point_msg.info.origin.orientation.z = ...
        nearest_point_msg.info.origin.orientation.w = ...
        nearest_point_msg.data = nearest_points.flatten()
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

