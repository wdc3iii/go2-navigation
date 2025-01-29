from typing import List, Optional

from rclpy.executors import SingleThreadedExecutor
from obelisk_py.core.utils.ros import spin_obelisk
import numpy as np
from obelisk_control_msgs.msg import PDFeedForward
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_control_msgs.msg import VelocityCommand
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound

import torch


class VelocityTrackingController(ObeliskController):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)
        # Velocity limits
        self.declare_parameter("v_x_max", 0.5)
        self.declare_parameter("v_y_max", 0.5)
        self.declare_parameter("w_z_max", 0.5)
        self.v_x_max = self.get_parameter("v_x_max").get_parameter_value().double_value
        self.v_y_max = self.get_parameter("v_y_max").get_parameter_value().double_value
        self.w_z_max = self.get_parameter("w_z_max").get_parameter_value().double_value

        # Load policy
        self.declare_parameter("policy_path", "")
        policy_path = self.get_parameter("policy_path").get_parameter_value().string_value
        self.policy = torch.load(policy_path)
        self.device = next(self.policy.parameters()).device

        # Set action scale, number of robot joints
        self.declare_parameter("action_scale", 0.25)
        self.action_scale = self.get_parameter("action_scale").get_parameter_value().double_value
        self.declare_parameter("num_motors", 12)
        self.num_motors = self.get_parameter("num_motors").get_parameter_value().integer_value

        # Set PD gains
        self.declare_parameter("kps", [25.] * self.num_motors)
        self.declare_parameter("kds", [0.5] * self.num_motors)
        self.kps = self.get_parameter("kps").get_parameter_value().double_array_value
        self.kds = self.get_parameter("kds").get_parameter_value().double_array_value

        # Phase info
        self.declare_parameter("phase_period", 0.3)
        self.phase_period = self.get_parameter("phase_period").get_parameter_value().double_value

        # Get default angles
        self.joint_names_isaac = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        self.joint_names_mujoco = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.declare_parameter("default_angles_isaac", [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])  # Default angles in IsaacSim order
        self.default_angles_isaac = np.array(self.get_parameter("default_angles_isaac").get_parameter_value().double_array_value)
        self.mujoco_to_isaac = [self.joint_names_mujoco.index(joint_name) for joint_name in self.joint_names_isaac]
        self.isaac_to_mujoco = [self.joint_names_isaac.index(joint_name) for joint_name in self.joint_names_mujoco]
        self.default_angles_mujoco = self.default_angles_isaac[self.isaac_to_mujoco]
        
        # Declare subscriber to velocity commands
        self.register_obk_subscription(
            "sub_high_level_ctrl_setting",
            self.joystick_callback,  # type: ignore
            key="sub_high_level_ctrl_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )

        self.get_logger().info(f"Policy: {policy_path} loaded on {self.device}. {len(self.kps)}, {len(self.kds)}")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)
        self.joint_pos = self.default_angles_mujoco.copy()
        self.joint_vel = np.zeros((self.num_motors,))
        self.cmd_vel = np.zeros((3,))
        self.proj_g = np.zeros((3,))
        self.proj_g[2] = -1
        self.omega = np.zeros((3,))
        self.phase = np.zeros((2,))
        self.zero_action = np.zeros((self.num_motors,))
        self.action = self.zero_action.tolist()
        self.t_start = None
        
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """
        if len(x_hat_msg.q_joints) == self.num_motors:
            self.joint_pos = np.array(x_hat_msg.q_joints)
        else:
            self.get_logger().error(f"Estimated State joint position size does not match URDF! Size is {len(x_hat_msg.q_joints)} instead of {self.num_motors}.")

        if len(x_hat_msg.v_joints) == self.num_motors:
            self.joint_vel = np.array(x_hat_msg.v_joints)
        else:
            self.get_logger().error(f"Estimated State joint velocity size does not match URDF! Size is {len(x_hat_msg.v_joints)} instead of {self.num_motors}.")

        if len(x_hat_msg.q_base) == 7:
            self.proj_g = self.project_gravity(x_hat_msg.q_base[3:7])
        else:
            self.get_logger().error(f"Estimated State base pose size does not match URDF! Size is {len(x_hat_msg.q_base)} instead of 7.")

        if len(x_hat_msg.v_base) == 6:
            self.omega = np.array(x_hat_msg.v_base[3:6])
        else:
            self.get_logger().error(f"Estimated State base velocity size does not match URDF! Size is {len(x_hat_msg.v_base)} instead of 6.")

        theta = 2 * np.pi / self.phase_period * (x_hat_msg.header.stamp.sec + x_hat_msg.header.stamp.nanosec * 1e-9)
        self.phase = np.array([np.cos(theta), np.sin(theta)])

    def joystick_callback(self, cmd_msg: VelocityCommand):
        self.cmd_vel[0] = cmd_msg.v_x
        self.cmd_vel[1] = cmd_msg.v_y
        self.cmd_vel[2] = cmd_msg.w_z

    @staticmethod
    def project_gravity(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]

        pg = np.zeros(3)

        pg[0] = 2 * (-qz * qx + qw * qy)
        pg[1] = -2 * (qz * qy + qw * qx)
        pg[2] = 1 - 2 * (qw * qw + qz * qz)

        return pg

    def compute_control(self) -> PDFeedForward:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Generate input to RL model
        obs = np.concatenate([
            self.omega,
            self.proj_g,
            self.cmd_vel,
            (self.joint_pos - self.default_angles_mujoco)[self.mujoco_to_isaac],
            self.joint_vel[self.mujoco_to_isaac],
            self.action,
            self.phase
        ])

        # Call RL model
        self.action = self.policy(torch.tensor(obs).to(self.device).float()).detach().cpu().numpy()

        # setting the message
        pd_ff_msg = PDFeedForward()
        pd_ff_msg.header.stamp = self.get_clock().now().to_msg()
        pos_targ = self.action[self.isaac_to_mujoco] * self.action_scale + self.default_angles_mujoco
        pd_ff_msg.pos_target = pos_targ.tolist()
        pd_ff_msg.vel_target = self.zero_action.tolist()
        pd_ff_msg.feed_forward = self.zero_action.tolist()
        pd_ff_msg.u_mujoco = np.concatenate([
            pos_targ,
            self.zero_action,
            self.zero_action
        ]).tolist()
        pd_ff_msg.joint_names = self.joint_names_mujoco
        pd_ff_msg.kp = self.kps
        pd_ff_msg.kd = self.kds
        self.obk_publishers["pub_ctrl"].publish(pd_ff_msg)
        assert is_in_bound(type(pd_ff_msg), ObeliskControlMsg)
        return pd_ff_msg
    

def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, VelocityTrackingController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()
