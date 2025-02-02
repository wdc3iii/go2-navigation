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

import casadi as ca
import torch
from go2_dyn_tube_mpc.casadi_utils import (
    quadratic_objective, dynamics_constraints, initial_condition_equality_constraint, setup_parameterized_ocp
)

import debugpy

debugpy.listen(("localhost", 5678))  # Port 5678 for debugging
print("Waiting for debugger...")
debugpy.wait_for_client()  # Pauses execution until debugger is attached

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
        self.declare_parameter("dt", 0.1)
        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.ts = np.arange(0, self.N + 1) * self.dt

        # Velocity bounds
        self.declare_parameter("v_max", [0.5, 0.5, 0.5])
        self.v_max_bound = np.array(self.get_parameter("v_max").get_parameter_value().double_array_value)
        self.declare_parameter("v_min", [-0.1, -0.5, -0.5])
        self.v_min_bound = np.array(self.get_parameter("v_min").get_parameter_value().double_array_value)
        assert np.all(self.v_max_bound > 0) and np.all(self.v_min_bound <= 0)
        # Cost matrices
        self.declare_parameter("Q", [1., 0., 0., 0., 1., 0., 0., 0., 0.1])          # State cost
        self.Q = np.array(self.get_parameter("Q").get_parameter_value().double_array_value).reshape(self.n, self.n)
        self.declare_parameter("Qf", [10., 0., 0., 0., 10., 0., 0., 0., 10.])       # Terminal Cost
        self.Qf = np.array(self.get_parameter("Qf").get_parameter_value().double_array_value).reshape(self.n, self.n)
        self.declare_parameter("R", [0.1, 0., 0., 0., 0.1, 0., 0., 0., 0.1])        # Temporal weights on state cost
        self.R = np.array(self.get_parameter("R").get_parameter_value().double_array_value).reshape(self.m, self.m)
        self.declare_parameter("Q_schedule", np.linspace(0, 1, self.N).tolist())    # Schedule on state cost (allow more deviation of trajectory from plan near beginning)
        self.Q_sched = np.array(self.get_parameter("Q_schedule").get_parameter_value().double_array_value)
        self.declare_parameter("Rv_first", [0., 0., 0., 0., 0., 0., 0., 0., 0.])      # First order rate penalty on input
        self.Rv_first = np.array(self.get_parameter("Rv_first").get_parameter_value().double_array_value).reshape(self.n, self.n)
        self.declare_parameter("Rv_second", [0., 0., 0., 0., 0., 0., 0., 0., 0.])     # Second order rate penalty on input
        self.Rv_second = np.array(self.get_parameter("Rv_second").get_parameter_value().double_array_value).reshape(self.n, self.n)

        # Declare subscriber to velocity commands
        self.register_obk_subscription(
            "sub_plan_setting",
            self.plan_callback,  # type: ignore
            key="sub_plan_key",  # key can be specified here or in the config file
            msg_type=Trajectory
        )
        self.register_obk_subscription(
            "sub_vel_lim_setting",
            self.vel_lim_callback,  # type: ignore
            key="sub_vel_lim_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        # TODO: DEBUGGIN
        # super().on_configure(state)
        self.n = 3
        self.m = 3
        self.px_z = np.zeros((3,))
        self.px_v = np.zeros((3,))
        self.z_ref = np.zeros((self.n, self.N + 1))
        self.v_ref = np.zeros((self.m, self.N))

        self.v_max = self.v_max_bound
        self.v_min = self.v_min_bound

        self.recieved_path = False

        setup_parameterized_ocp(
            self.N, self.n, self.m, self.Q, self.R, self.Qf, self.Q_sched, self.Rv_first, self.Rv_second
        )
        
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """

        if len(x_hat_msg.q_base) == 7:
            self.px_z = np.array([
                x_hat_msg.q_base[0],                    # x
                x_hat_msg.q_base[1],                    # y
                self.quat2yaw(x_hat_msg.q_base[3:])     # theta
            ])
        else:
            self.get_logger().error(f"Estimated State base pose size does not match URDF! Size is {len(x_hat_msg.q_base)} instead of 7.")

        if len(x_hat_msg.v_base) == 6:
            self.px_v = np.array([
                x_hat_msg.v_base[0],                    # v_x
                x_hat_msg.v_base[1],                    # v_y
                x_hat_msg.v_base[-1]                    # w_z
            ])
        else:
            self.get_logger().error(f"Estimated State base velocity size does not match URDF! Size is {len(x_hat_msg.v_base)} instead of 6.")

    def plan_callback(self, plan_msg: Trajectory):
        # Accepts a trajectory which is a sequence of waypoints
        self.plan = np.array(plan_msg.z).reshape(plan_msg.horizon + 1, plan_msg.n)
        self.recieved_path = True

    def vel_lim_callback(self, v_max_msg: VelocityCommand):
        """Sets the upper bound on velocity limits for the planner"""
        prop_v_lim = np.clip(np.array(v_max_msg.v_x, v_max_msg.v_y, v_max_msg.w_z), 0, 1)
        self.v_max = prop_v_lim * self.v_max_bound
        self.v_min = prop_v_lim * self.v_min_bound

    @staticmethod
    def quat2yaw(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw

    def compute_control(self) -> Trajectory:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Don't plan if we haven't received a path to follow
        if not self.recieved_path:
            return
        
        z, v = self.solve_dtmpc()

        # setting the message
        traj_msg = Trajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.z = z.tolist()
        traj_msg.v = v.tolist()
        traj_msg.t = self.ts.tolist()
        
        self.obk_publishers["pub_ctrl"].publish(traj_msg)
        assert is_in_bound(type(traj_msg), ObeliskControlMsg)
        return traj_msg

    def solve_dtmpc(self):
        # TODO: solve DTMPC
        # Compute z_ref, v_ref from path
        

        # Linearize obstacle constraints

        # Take a few steps on SQP
        z = ...
        v = ...
        return z, v
    
    



def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, DynamicTubeMPCNode, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
