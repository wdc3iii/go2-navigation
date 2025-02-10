from scripts.dynamic_tube_mpc_sdf import DynamicTubeMPC
import numpy as np
import matplotlib.pyplot as plt
from go2_dyn_tube_mpc.go2_dyn_tube_mpc.high_level_planner import HighLevelPlanner
import time

if __name__ == "__main__":
    # Create DTMPC
    N = 20
    n = 3
    m = 3
    dt = 0.1
    res = 0.05
    v_max = np.array([0.5, 0.5, 1.5])
    v_min = np.array([-0.1, -0.5, -1.5])
    Q = np.array([1., 1., 0.1])
    Qf = np.array([10., 10., 10.])
    R = np.array([0.05, 0.01, 0.01])
    Q_sched = np.linspace(0, 1, N)
    Rv_first = np.array([0., 0., 0.])
    Rv_second = np.array([0., 0., 0.])

    dtmpc = DynamicTubeMPC(dt, N, n, m, Q, Qf, R, Rv_first, Rv_second, Q_sched, v_min, v_max, robot_radius=0.15, sdf_size=200, sdf_resolution=res, obs_constraint_method="Constraint")
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

    dtmpc.update_sdf(occ_grid, np.array([0, 0, 0]))

    z_i = np.array([0.25, 0.25, 0.])
    dtmpc.set_initial_condition(z_i)
    z_f = np.array([1.75, 1.75, 0.])

    # Create graph solve
    explorer = HighLevelPlanner(3, free=0, uncertain=1, occupied=2)
    explorer.set_map(occ_grid, (0, 0, 0), res)

    start = explorer.map.pose_to_map(z_i[:2])
    goal = explorer.map.pose_to_map(z_f[:2])
    path, dist, frontiers = explorer.find_frontiers_to_goal(start, goal)
    path_t = explorer.map.map_to_pose(np.array(path))
    dtmpc.set_path(path_t)

    z_all = []
    v_all = []
    for _ in range(100):
        z_sol, v_sol = dtmpc.solve()
        # _, ax = plt.subplots(1, 2)
        # ax[0].plot(np.arange(dtmpc.N + 1) * dtmpc.dt, z_sol)
        # ax[0].plot(np.arange(dtmpc.N + 1) * dtmpc.dt, dtmpc.z_ref, '--')
        # ax[0].legend(['x', 'y', 'yaw', 'xd', 'yd', 'yawd'])
        #
        # ax[1].plot(np.arange(dtmpc.N) * dtmpc.dt, v_sol)
        # ax[1].plot(np.arange(dtmpc.N ) * dtmpc.dt, dtmpc.v_ref, '--')
        # ax[1].legend(['vx', 'vy', 'wz', 'vxd', 'vyd', 'wzd'])
        # plt.show()
        z_all.append(z_sol[0])
        v_all.append(v_sol[0])

        fig, ax = plt.subplots()
        dtmpc.map.plot(ax=ax)
        plt.plot(path_t[:, 0] / dtmpc.map.resolution, path_t[:, 1] / dtmpc.map.resolution, 'r')
        plt.plot(z_sol[:, 0] / dtmpc.map.resolution, z_sol[:, 1] / dtmpc.map.resolution, '.-b')
        plt.show()
        dtmpc.set_initial_condition(z_sol[1])
        time.sleep(0.1)

    # Plot everything
    z_all = np.array(z_all)
    v_all = np.array(v_all)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(z_all.shape[0]) * dtmpc.dt, z_all)
    ax[0].plot(np.arange(path_t.shape[0]) * dtmpc.dt, path_t, '--')
    ax[0].legend(['x', 'y', 'yaw', 'xd', 'yd'])

    ax[1].plot(np.arange(v_all.shape[0]) * dtmpc.dt, v_all)
    ax[1].legend(['vx', 'vy', 'wz'])
    plt.show()

    fig, ax = plt.subplots()
    explorer.map.plot(ax=ax)
    plt.plot(path_t[:, 0] / explorer.map.resolution, path_t[:, 1] / explorer.map.resolution, 'r')
    plt.plot(z_all[:, 0] / explorer.map.resolution, z_all[:, 1] / explorer.map.resolution, '.-b')
    plt.show()
