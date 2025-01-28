from typing import Type

import numpy as np
from obelisk_control_msgs.msg import PDFeedForward
from obelisk_estimator_msgs.msg import EstimatedState
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound


class VelocityTrackingController(ObeliskController):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)
        self.declare_parameter("v_x_max", 0.5)
        self.declare_parameter("v_y_max", 0.5)
        self.declare_parameter("w_z_max", 0.5)
        # self.get_logger().info(f"test_param: {self.get_parameter('test_param').get_parameter_value().string_value}")

        # Declare subscriber to velocity commands

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)
        self.joint_pos = None
        self.joint_vel = None
        self.cmd_vel = None
        self.quat = None
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: Type) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """
        # Update from callback
        pass  # do nothing

    def compute_control(self) -> Type:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # computing the control input
        u = np.sin(self.t)  # example state-independent control input

        # setting the message
        pd_feedforward_msg = PDFeedForward()
        pd_feedforward_msg.u_mujoco = [u]
        pd_feedforward_msg.q_des = [u]
        pd_feedforward_msg.qd_des = 0
        pd_feedforward_msg.tau_des = 0
        self.obk_publishers["pub_ctrl"].publish(pd_feedforward_msg)
        assert is_in_bound(type(pd_feedforward_msg), ObeliskControlMsg)
        return pd_feedforward_msg  # type: ignore