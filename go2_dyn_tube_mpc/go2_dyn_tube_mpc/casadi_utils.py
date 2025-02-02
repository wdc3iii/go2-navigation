import casadi as ca
import numpy as np


def lateral_unicycle_dynamics(z, v):
    gv_0 = v[0] * ca.cos(z[2]) - v[1] * ca.sin(z[2])
    gv_1 = v[0] * ca.sin(z[2]) + v[1] * ca.cos(z[2])
    gv_2 = v[2]
    return z + ca.vertcat(gv_0, gv_1, gv_2)

def quadratic_objective(x, Q, goal=None, t_sched=None):
    if goal is None:
        dist = x
    else:
        if goal.shape == x.shape:
            dist = x - goal
        else:
            dist = x - ca.repmat(goal, x.shape[0], 1)
    obj = ca.sum2((dist @ Q) * dist)
    if t_sched is not None:
        obj *= t_sched
    return ca.sum1(obj)

def dynamics_constraints(z, v):
    g_dyn = []
    for k in range(v.shape[0]):
        g_dyn.append(lateral_unicycle_dynamics(z[k, :].T, v[k, :].T).T - z[k + 1, :])
    g_dyn = ca.horzcat(*g_dyn)
    g_dyn_lb = ca.DM(*g_dyn.shape)
    g_dyn_ub = ca.DM(*g_dyn.shape)

    return g_dyn, g_dyn_lb, g_dyn_ub

def initial_condition_equality_constraint(z, z0):
    dist = z - z0
    return dist, ca.DM(*dist.shape), ca.DM(*dist.shape)


def setup_parameterized_ocp(N, n, m, Q, R, Qf, Q_sched, Rv_first, Rv_second, ):
        # No state constraints
        z_lb = ca.DM(N + 1, n)
        z_lb[:] = -ca.inf
        z_ub = ca.DM(N + 1, n)
        z_ub[:] = ca.inf

        def lbx(v_min):
            v_lb = ca.DM(np.repeat(v_min[:, None], N, 0))
            return ca.vertcat(
                ca.reshape(z_lb, (N + 1) * n, 1),
                ca.reshape(v_lb, N * m, 1),
            )
        
        def ubx(v_max):
            v_ub = ca.DM(np.repeat(v_max[:, None], N, 0))
            return ca.vertcat(
                ca.reshape(z_ub, (N + 1) * n, 1),
                ca.reshape(v_ub, N * m, 1),
            )

        # State variables
        z = ca.MX.sym("z", N + 1, n)
        v = ca.MX.sym("v", N, m)

        # Parameters for z_ref
        p_z_ref = ca.MX.sym("p_z_ref", N + 1, n)
        p_v_ref = ca.MX.sym("p_v_ref", N, m)

        # Initial condition parameter
        p_z0 = ca.MX.sym("p_z0", 1, n)  # Initial projection Pz(x0) state

        # Define cost function
        # Reference Tracking
        obj = quadratic_objective(z[:-1, :], Q, goal=p_z_ref[:-1, :], t_sched=Q_sched) \
            + quadratic_objective(v, R, goal=p_v_ref) \
            + quadratic_objective(z[-1, :], Qf, goal=p_z_ref[-1, :])

        # Smoothness of input
        # First order difference
        if np.any(Rv_first > 0):
            Rv_first = ca.DM(Rv_first)
            obj += quadratic_objective(v[:-1, :] - v[1:, :], Rv_first)
        # Second order difference
        if np.any(Rv_second > 0):
            Rv_second = ca.DM(Rv_second)
            first = v[:-1, :] - v[1:, :]
            obj += quadratic_objective(first[:-1, :] - first[1:, :], Rv_second)

        # Define dynamics constraint
        g_dyn, g_dyn_lb, g_dyn_ub = dynamics_constraints(z, v)
        g_ic, g_lb_ic, g_ub_ic = initial_condition_equality_constraint(z[0, :], p_z0)
        # TODO: define obstacle constraints (or costs? or both?)
        # g_tube, g_lb_tube, g_ub_tube = tube_dynamics(z, v, w, e, v_prev)
        
        # TODO: define tube contraint

        g = ca.horzcat(g_dyn, g_ic)
        g_lb = ca.horzcat(g_dyn_lb, g_lb_ic)
        g_ub = ca.horzcat(g_dyn_ub, g_ub_ic)
        g = g.T
        g_lb = g_lb.T
        g_ub = g_ub.T

        # Generate solver
        x_nlp = ca.vertcat(
            ca.reshape(z, (N + 1) * n, 1),
            ca.reshape(v, N * m, 1),
        )
        p_nlp = ca.vertcat(
            p_z0.T, 
            ca.reshape(p_z_ref, (N + 1) * n, 1), ca.reshape(p_v_ref, N * m, 1),
        )

        # x_cols, g_cols, p_cols = generate_col_names(N, x_nlp, g, p_nlp)
        nlp_dict = {
            "x": x_nlp,
            "f": obj,
            "g": g,
            "p": p_nlp
        }
        nlp_opts = {
            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-4,
            # "ipopt.print_level": 5,
            "print_time": True,
        }

        nlp_solver = ca.nlpsol("trajectory_generator", "ipopt", nlp_dict, nlp_opts)

        solver = {
            "solver": nlp_solver, "lbg": g_lb, "params": p_nlp,
            "ubg": g_ub, "lbx": lbx, "ubx": ubx
        } #, "g_cols": g_cols, "x_cols": x_cols, "p_cols": p_cols}
        return solver


def init_params(z0, z_cost, v_cost):

    params = np.vstack([
        z0[:, None],
        np.reshape(z_cost, (-1, 1), order='F'), 
        np.reshape(v_cost, (-1, 1), order='F')
    ])
    return params


def init_decision_var(z, v):
    N = v.shape[0]
    n = z.shape[1]
    m = v.shape[1]
    
    x_init = np.vstack([
        np.reshape(z, ((N + 1) * n, 1), order='F'),
        np.reshape(v, (N * m, 1), order='F')
    ])
    return x_init


def extract_solution(sol, N, n, m):
    z_ind = (N + 1) * n
    v_ind = N * m
    z_sol = np.array(sol["x"][:z_ind, :].reshape((N + 1, n)))
    v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((N, m)))
    return z_sol, v_sol


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

    solver = setup_parameterized_ocp(N, n, m, Q, R, Qf, Q_sched, Rv_first, Rv_second)

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

    params = init_params(z_i, z_ref, v_ref)
    x_init = init_decision_var(z_init, v_init)

    sol = solver["solver"](
        x0=x_init,
        p=params,
        lbg=solver["lbg"],
        ubg=solver["ubg"],
        lbx=solver["lbx"](v_min),
        ubx=solver["ubx"](v_max)
    )

    z_sol, v_sol = extract_solution(sol, N, n, m)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(z_sol)
    plt.plot(z_ref, '--')
    plt.legend(['x', 'y', 'yaw', 'xd', 'yd', 'yawd'])
    plt.savefig("debug_plot_state.png")  # Saves the figure as an image

    plt.figure()
    plt.plot(v_sol)
    plt.plot(v_ref, '--')
    plt.legend(['vx', 'vy', 'wz', 'vxd', 'vyd', 'wzd'])
    plt.savefig("debug_plot_input.png")  # Saves the figure as an image    

    print('here')