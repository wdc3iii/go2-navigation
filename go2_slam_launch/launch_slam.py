import os
import time
import subprocess
from launch.actions import SetEnvironmentVariable
from launch import LaunchDescription
from launch_ros.actions import Node

# REMOTE_HOST = "unitree@192.168.123.18"  # Change to your remote computer's username and hostname/IP
# REMOTE_LAUNCH_CMD = "source ~/dev_ws/install/setup.bash && " \
#                     "source /unitree/module/graph_pid_ws/install/setup.bash && " \
#                     "export ROS_DOMAIN_ID=5 && " \
#                     "cd dev_ws/ && ros2 launch launch/launch_lidar_and_odom.py"

# def launch_remote():
#     """Launch a ROS 2 node on a remote machine using SSH."""
#     ssh_command = f"ssh {REMOTE_HOST} '{REMOTE_LAUNCH_CMD}'"
#     subprocess.Popen(ssh_command, shell=True)

def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable("ROS_DOMAIN_ID", "5"),
        Node(
            package="slam_toolbox",
            executable="sync_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[{
                "odom_frame": "odom",
                "map_frame": "map",
                "base_frame": "base_link",
                "scan_topic": "/scan",
                "odom_topic": "/rs_t265/odom",
                "use_sim_time": False
            }]
        )
    ])
