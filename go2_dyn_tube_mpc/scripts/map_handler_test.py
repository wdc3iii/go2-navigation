import matplotlib.pyplot as plt

from go2_dyn_tube_mpc.go2_dyn_tube_mpc.high_level_planner import HighLevelPlanner
import numpy as np

FREE = 0
UNCERTAIN = 1
OCCUPIED = 2


# ROS map is [y, x]
# Create a map, resolution = 0.05, which has a single rectangle of occupied territory in the bottom left,
# and a single rectangle of uncertain territory in the top right
width = 1.5
height = 1
res = 0.05
ros_map = np.zeros((int(height / res), int(width / res)))

occ_x_min = 0.1
occ_x_max = 1.1
occ_y_min = 0.2
occ_y_max = 0.375

x1 = int(occ_x_min / res)
x2 = int(occ_x_max / res)
y1 = int(occ_y_min / res)
y2 = int(occ_y_max / res)
ros_map[y1:y2, x1:x2] = OCCUPIED

occ_x_min = 0.55
occ_x_max = 0.9
occ_y_min = 0.7
occ_y_max = 0.8

x1 = int(occ_x_min / res)
x2 = int(occ_x_max / res)
y1 = int(occ_y_min / res)
y2 = int(occ_y_max / res)
ros_map[y1:y2, x1:x2] = UNCERTAIN

robot_radius = 0.15
map_handler = HighLevelPlanner(robot_radius=round(robot_radius / res), free=FREE, uncertain=UNCERTAIN, occupied=OCCUPIED)
map_handler.set_map(ros_map, np.array([0, 0, 0]), res)

_, ax = plt.subplots()
map_handler.map.plot(ax)
plt.show()

_, ax = plt.subplots()
map_handler.inflated_map.plot(ax)
plt.show()

# map_handler.update_origin(np.array([-0.75, 0., np.pi / 4]))
# _, ax = plt.subplots()
# map_handler.map.plot(ax)
# ax.set_xlim([-1, 1])
# ax.set_ylim([0, 2])
# plt.show()


# Check pose to map and map to pose
# update origin, x translation
p = np.array([[0, 0], [1, 0.], [0., 1.]])
inds = np.array([[0, 0], [10, 0], [0, 10]])
print(p, " -> ", map_handler.map.pose_to_map(p), "\n")
print(inds, " -> ", map_handler.map.map_to_pose(inds), "\n")

print("Move origin left 1m\n")
map_handler.update_origin(np.array([-1., 0., 0]))
print(p, " -> ", map_handler.map.pose_to_map(p), "\n")
print(inds, " -> ", map_handler.map.map_to_pose(inds), "\n")

print("Move origin down 1m\n")
map_handler.update_origin(np.array([0., -1., 0]))
print(p, " -> ", map_handler.map.pose_to_map(p), "\n")
print(inds, " -> ", map_handler.map.map_to_pose(inds), "\n")

print("Rotate origin -45deg\n")
map_handler.update_origin(np.array([0., 0., -np.pi / 4]))
print(p, " -> ", map_handler.map.pose_to_map(p), "\n")
print(inds, " -> ", map_handler.map.map_to_pose(inds), "\n")

print("Rotate origin -45deg, left 1m, down 1m\n")
map_handler.update_origin(np.array([-1, -1., -np.pi / 4]))
print(p, " -> ", map_handler.map.pose_to_map(p), "\n")
print(inds, " -> ", map_handler.map.map_to_pose(inds), "\n")