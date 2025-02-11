from go2_dyn_tube_mpc.dynamic_tube_mpc import DynamicTubeMPC
import numpy as np
import matplotlib.pyplot as plt
from go2_dyn_tube_mpc.high_level_planner import HighLevelPlanner
import time
import threading

visualize = True

if __name__ == "__main__":
    # Create DTMPC
    N = 20
    n = 3
    m = 3
    dt = 0.1
    res = 0.05
    robot_radius = 0.15
    v_max = np.array([0.5, 0.5, 1.5])
    v_min = np.array([-0.1, -0.5, -1.5])
    Q = np.array([1., 1., 0.1])
    Qf = np.array([10., 10., 10.])
    R = np.array([0.05, 0.01, 0.01])
    Q_sched = np.linspace(0, 1, N)
    Rv_first = np.array([0., 0., 0.])
    Rv_second = np.array([0., 0., 0.])

    dtmpc = DynamicTubeMPC(dt, N, n, m, Q, Qf, R, Rv_first, Rv_second, Q_sched, v_min, v_max, robot_radius=robot_radius, obs_constraint_method="Constraint")
    dtmpc.set_input_bounds(v_min, v_max)
    # Create problem
    occ_grid = np.zeros((41, 41))
    # index map via [y x]
    occ_grid[10:15, :30] = 2
    occ_grid[25:30, 10:] = 1
    occ_grid[0, :] = 1
    occ_grid[-1, :] = 1
    occ_grid[:, 0] = 1
    occ_grid[:, -1] = 1

    scan = np.array([
        [-0.2, -0.2],
        [-0.2, -0.3],
        [-0.2, -0.4],
        # [0.2, 0.11],
        # [0.2, 0.12],
        # [0.2, 0.13],
        # [0.225, 0.14],
        # [0.25, 0.15],
        # [0.4, 0.65],
        # [0.41, 0.65],
        # [0.42, 0.65],
        # [0.43, 0.65],
        # [0.44, 0.65],
        # [0.45, 0.65],
        # [0.8, 0.9],
        # [0.82, 0.9],
        # [0.84, 0.9],
        # [0.86, 0.9],
        # [0.88, 0.9],
        # [0.9, 0.9],
    ])

    dtmpc.update_scan(scan)

    z_i = np.array([0.3, 0.3, 0.])
    dtmpc.set_initial_condition(z_i)
    z_f = np.array([1.8, 1.8, 0.])

    # Create graph solve
    explorer = HighLevelPlanner(threading.Lock(), threading.Lock(),3, robot_radius=round(robot_radius / res), free=0, uncertain=1, occupied=2)
    explorer.set_map(occ_grid, (0, 0, 0), res)

    start = explorer.map.pose_to_map(z_i[:2])
    goal = explorer.map.pose_to_map(z_f[:2])
    path, dist, frontiers, info = explorer.find_frontiers_to_goal(start, goal)
    dtmpc.set_path(path)

    nearest_inds, nearest_dists, sub_map_origin, sub_map_yaw = explorer.compute_nearest_inds(z_i[:2], 100)
    dtmpc.update_nearest_inds(nearest_inds, nearest_dists, sub_map_origin, sub_map_yaw)

    dtmpc.reset_warm_start()
    z_all = []
    v_all = []
    for _ in range(125):
        nearest_inds, nearest_dists, sub_map_origin, sub_map_yaw = explorer.compute_nearest_inds(dtmpc.z0[:2], 30)
        dtmpc.update_nearest_inds(nearest_inds, nearest_dists, sub_map_origin, sub_map_yaw)
        z_sol, v_sol = dtmpc.solve()
        z_all.append(z_sol[0])
        v_all.append(v_sol[0])

        if visualize:
            fig, ax = plt.subplots()
            explorer.map.plot(ax=ax)
            plt.plot(path[:, 0], path[:, 1], 'r')
            plt.plot(z_sol[:, 0], z_sol[:, 1], '.-b')
            plt.show()
            time.sleep(0.1)

        dtmpc.set_initial_condition(z_sol[1])

    # Plot everything
    z_all = np.array(z_all)
    v_all = np.array(v_all)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(z_all.shape[0]) * dtmpc.dt, z_all)
    ax[0].plot(np.arange(path.shape[0]) * dtmpc.dt, path, '--')
    ax[0].legend(['x', 'y', 'yaw', 'xd', 'yd'])

    ax[1].plot(np.arange(v_all.shape[0]) * dtmpc.dt, v_all)
    ax[1].legend(['vx', 'vy', 'wz'])
    plt.show()

    fig, ax = plt.subplots()
    explorer.map.plot(ax=ax)
    plt.plot(path[:, 0], path[:, 1], 'r')
    plt.plot(z_all[:, 0], z_all[:, 1], '.-b')
    plt.show()
