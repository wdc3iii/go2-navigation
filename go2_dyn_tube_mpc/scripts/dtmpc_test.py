from go2_dyn_tube_mpc.dynamic_tube_mpc import DynamicTubeMPC
import numpy as np

if __name__ == "__main__":
    N = 20
    n = 3
    m = 3
    dt = 0.1
    v_max = np.array([0.5, 0.5, 0.5])
    v_min = np.array([-0.1, -0.5, -0.5])
    Q = np.array([1., 0., 0., 0., 1., 0., 0., 0., 0.1]).reshape(n, n)
    Qf = np.array([10., 0., 0., 0., 10., 0., 0., 0., 10.]).reshape(n, n)
    R = np.array([0.1, 0., 0., 0., 0.1, 0., 0., 0., 0.1]).reshape(m, m)
    Q_sched = np.linspace(0, 1, N)
    Rv_first = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(n, n)
    Rv_second = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(n, n)

    dtmpc = DynamicTubeMPC(N, n, m, Q, Qf, R, Rv_first, Rv_second, Q_sched)

    tt = np.arange(0, N + 1)[:, None] * dt
    z_ref = np.hstack([
        np.sin(tt),
        np.cos(tt),
        tt 
    ]) * v_max
    v_ref = np.hstack([
        np.ones((N, 1)),
        np.zeros((N, 1)),
        np.zeros((N, 1))
    ]) * v_max
    z_i = np.array([0., 1., 0.])
    # z_ref = np.hstack([
    #     np.arange(0, N + 1)[:, None] * dt,
    #     np.zeros((N + 1, 1)),
    #     np.zeros((N + 1, 1))
    # ]) * v_max
    # v_ref = np.hstack([
    #     np.ones((N, 1)),
    #     np.zeros((N, 1)),
    #     np.zeros((N, 1))
    # ]) * v_max
    # z_i = np.array([0., 0., 0.])
    z_init = np.zeros_like(z_ref)
    v_init = np.zeros_like(v_ref)

    dtmpc.set_input_bounds(v_min, v_max)
    dtmpc.set_reference(z_ref, v_ref)
    dtmpc.set_initial_condition(z_i)
    dtmpc.set_warm_start(z_init, v_init)

    z_sol, v_sol = dtmpc.solve()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(z_sol)
    plt.plot(z_ref, '--')
    plt.legend(['x', 'y', 'yaw', 'xd', 'yd', 'yawd'])
    plt.show()

    plt.figure()
    plt.plot(v_sol)
    plt.plot(v_ref, '--')
    plt.legend(['vx', 'vy', 'wz', 'vxd', 'vyd', 'wzd'])
    plt.show()

    print('here')