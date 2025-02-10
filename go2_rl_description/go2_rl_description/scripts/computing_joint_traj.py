import mujoco
import glfw
import numpy as np

# Load the XML model
model = mujoco.MjModel.from_xml_path("../mujoco/go2_fixed.xml")
data = mujoco.MjData(model)

# Initialize GLFW for visualization
glfw.init()
window = glfw.create_window(1200, 800, "MuJoCo Viewer", None, None)
glfw.make_context_current(window)

# Create visualization objects
scene = mujoco.MjvScene(model, maxgeom=10000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

# Set camera properties
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# 1. Load keyframe and set the robot state
keyframe_id = 0  # Assuming the keyframe ID you want to use is 0
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
mujoco.mj_forward(model, data)  # Perform forward kinematics
print("Loaded keyframe:", keyframe_id)

# 2. Compute forward kinematics to determine the positions of the feet
foot_names = ["FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site"]  # Replace with the actual names in your model
foot_positions = {}
foot_ids = []
default_foot_pos = []
for foot in foot_names:
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot)
    foot_ids.append(foot_id)
    site_pos = data.site_xpos[foot_id]
    foot_positions[foot_id] = site_pos.copy()
    print(f"Position of {foot}: {site_pos}")


def cubic_bezier_interpolation(z_start, z_end, t):
    t = np.clip(t, 0, 1)
    z_diff = z_end - z_start
    bezier = t ** 3 + 3 * (t ** 2 * (1 - t))
    return z_start + z_diff * bezier

swing_height = 0.08
def joint_traj(t):
    z = np.where(
        t <= 0.5,
        cubic_bezier_interpolation(0, swing_height, 2 * t),  # Upward portion of swing
        cubic_bezier_interpolation(swing_height, 0, 2 * t - 1)  # downward portion of swing
    )
    return np.array([0., 0., z])


# 3. Inverse kinematics for generating joint trajectories
def inverse_kinematics(t):
    for foot_id in foot_ids:
        target_pos = foot_positions[foot_id] + joint_traj(t)
        if foot_id == foot_ids[-1]:
            print(f"{t}: {target_pos}")
        """Simple iterative IK solver using gradient descent."""
        for i in range(1000):  # Maximum iterations
            mujoco.mj_forward(model, data)  # Perform forward kinematics
            current_position = data.site_xpos[foot_id]
            error = target_pos - current_position
            if np.linalg.norm(error) < 1e-9:
                break
            # Compute joint corrections using pseudo-inverse Jacobian
            jacobian_pos = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacobian_pos, None, foot_id)
            delta_q = np.linalg.pinv(jacobian_pos) @ error
            data.qpos += delta_q
    return data.qpos

# Target foot trajectory (up-down movement)
time_steps = 101
ts = np.linspace(0, 1, time_steps)
z_des = []
z_act = []

joint_trajectory = np.zeros((time_steps, model.nv))
# Generate and visualize the inverse kinematics results
for ind, t in enumerate(ts):
    data.qpos[:] = inverse_kinematics(t)
    joint_trajectory[ind] = data.qpos.copy()
    mujoco.mj_forward(model, data)

    # Update scene and render
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 1200, 800), scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

joint_names_isaac = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
]
joint_names_mujoco = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]

# np.savetxt("joint_trajectory.txt", joint_trajectory[:, :3])

import matplotlib.pyplot as plt
plt.plot(joint_trajectory)
plt.show()
mujoco_to_isaac = [joint_names_mujoco.index(joint_name) for joint_name in joint_names_isaac]
isaac_to_mujoco = [joint_names_isaac.index(joint_name) for joint_name in joint_names_mujoco]
np.savetxt("joint_trajectory.txt", joint_trajectory[:, mujoco_to_isaac])
plt.plot(joint_trajectory[:, mujoco_to_isaac])
plt.show()