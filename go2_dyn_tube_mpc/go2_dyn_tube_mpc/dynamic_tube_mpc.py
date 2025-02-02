import numpy as np
import casadi as ca

class DynamicTubeMPC:
    
    def __init__(self, N, n, m, Q, Qf, R, Rv_f, Rv_s, Q_sched):
        # Dimensions
        self.N = N
        self.n = n
        self.m = m

        # Cost function parameters
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.Rv_f = Rv_f
        self.Rv_s = Rv_s
        self.Q_sched = Q_sched

        # Input bounds
        self.v_max = np.zeros((m,))
        self.v_min = np.zeros((m,))

        # Reference Trajectory
        self.z_ref = np.zeros((N + 1, n))
        self.v_ref = np.zeros((N, m))

        # Initial condition
        self.z0 = np.zeros((n,))

        # Warm starts
        self.z_warm = np.zeros((N + 1, n))
        self.v_warm = np.zeros((N, m))

        # Map / scan
        self.map = None
        self.occupied = set()


        # Casadi variables
        self.z_lb = None
        self.z_ub = None
        self.g = None
        self.g_lb = None
        self.g_ub = None
        self.obj = None
        self.x_nlp = None
        self.p_nlp = None
        self.nlp_solver = None
        self.nlp_opts = {}
        self.setup_ocp()

    def solve(self):
        params = self.init_params()
        x_init = self.init_decision_var()
        sol = self.nlp_solver(
            x0=x_init, p=params, lbg=self.g_lb, ubg=self.g_ub, lbx=self.lbx(),
                         ubx=self.ubx()
        )
        return self.extract_solution(sol)

    def set_input_bounds(self, v_min, v_max):
        assert v_min.shape == self.v_min.shape and v_max.shape == self.v_max.shape
        self.v_min = v_min
        self.v_max = v_max

    def set_reference(self, z_ref, v_ref):
        assert z_ref.shape == self.z_ref.shape and v_ref.shape == self.v_ref.shape
        self.z_ref = z_ref
        self.v_ref = v_ref

    def set_initial_condition(self, z0):
        assert z0.shape == self.z0.shape
        self.z0 = z0

    def set_warm_start(self, z_warm, v_warm):
        assert z_warm.shape == self.z_warm.shape and v_warm.shape == self.v_warm.shape
        self.z_warm = z_warm
        self.v_warm = v_warm

    def lbx(self):
        v_lb = ca.DM(np.repeat(self.v_min[:, None], self.N, 0))
        return ca.vertcat(
            ca.reshape(self.z_lb, (self.N + 1) * self.n, 1),
            ca.reshape(v_lb, self.N * self.m, 1),
        )
    
    def ubx(self):
        v_ub = ca.DM(np.repeat(self.v_max[:, None], self.N, 0))
        return ca.vertcat(
            ca.reshape(self.z_ub, (self.N + 1) * self.n, 1),
            ca.reshape(v_ub, self.N * self.m, 1),
        )

    def init_params(self):
        params = np.vstack([
            self.z0[:, None],
            np.reshape(self.z_ref, (-1, 1), order='F'),
            np.reshape(self.v_ref, (-1, 1), order='F')
        ])
        return params

    def init_decision_var(self):
        x_init = np.vstack([
            np.reshape(self.z_warm, ((self.N + 1) * self.n, 1), order='F'),
            np.reshape(self.v_warm, (self.N * self.m, 1), order='F')
        ])
        return x_init

    def extract_solution(self, sol):
        z_ind = (self.N + 1) * self.n
        v_ind = self.N * self.m
        z_sol = np.array(sol["x"][:z_ind, :].reshape((self.N + 1, self.n)))
        v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((self.N, self.m)))
        return z_sol, v_sol

    @staticmethod
    def lateral_unicycle_dynamics(z, v):
        gv_0 = v[0] * ca.cos(z[2]) - v[1] * ca.sin(z[2])
        gv_1 = v[0] * ca.sin(z[2]) + v[1] * ca.cos(z[2])
        gv_2 = v[2]
        return z + ca.vertcat(gv_0, gv_1, gv_2)

    @staticmethod
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

    def dynamics_constraints(self, z, v):
        g_dyn = []
        for k in range(v.shape[0]):
            g_dyn.append(self.lateral_unicycle_dynamics(z[k, :].T, v[k, :].T).T - z[k + 1, :])
        g_dyn = ca.horzcat(*g_dyn)
        g_dyn_lb = ca.DM(*g_dyn.shape)
        g_dyn_ub = ca.DM(*g_dyn.shape)

        return g_dyn, g_dyn_lb, g_dyn_ub

    @staticmethod
    def equality_constraint(z, z0):
        dist = z - z0
        return dist, ca.DM(*dist.shape), ca.DM(*dist.shape)

    def setup_ocp(self):
        self.z_lb = ca.DM(self.N + 1, self.n)
        self.z_lb[:] = -ca.inf
        self.z_ub = ca.DM(self.N + 1, self.n)
        self.z_ub[:] = ca.inf

        # State variables
        z = ca.MX.sym("z", self.N + 1, self.n)
        v = ca.MX.sym("v", self.N, self.m)

        # Parameters for z_ref
        p_z_ref = ca.MX.sym("p_z_ref", self.N + 1, self.n)
        p_v_ref = ca.MX.sym("p_v_ref", self.N, self.m)

        # Initial condition parameter
        p_z0 = ca.MX.sym("p_z0", 1, self.n)  # Initial projection Pz(x0) state

        # Define cost function
        # Reference Tracking
        self.obj = self.quadratic_objective(z[:-1, :], self.Q, goal=p_z_ref[:-1, :], t_sched=self.Q_sched) \
              + self.quadratic_objective(v, self.R, goal=p_v_ref) \
              + self.quadratic_objective(z[-1, :], self.Qf, goal=p_z_ref[-1, :])

        # Smoothness of input
        # First order difference
        if np.any(self.Rv_f > 0):
            Rv_f_ = ca.DM(self.Rv_f)
            self.obj += self.quadratic_objective(v[:-1, :] - v[1:, :], Rv_f_)
        # Second order difference
        if np.any(self.Rv_s > 0):
            Rv_s_ = ca.DM(self.Rv_s)
            first = v[:-1, :] - v[1:, :]
            self.obj += self.quadratic_objective(first[:-1, :] - first[1:, :], Rv_s_)

        # Define dynamics constraint
        g_dyn, g_dyn_lb, g_dyn_ub = self.dynamics_constraints(z, v)
        g_ic, g_lb_ic, g_ub_ic = self.equality_constraint(z[0, :], p_z0)
        # TODO: define obstacle constraints (or costs? or both?)
        # g_tube, g_lb_tube, g_ub_tube = tube_dynamics(z, v, w, e, v_prev)

        # TODO: define tube contraint

        g = ca.horzcat(g_dyn, g_ic)
        g_lb = ca.horzcat(g_dyn_lb, g_lb_ic)
        g_ub = ca.horzcat(g_dyn_ub, g_ub_ic)
        self.g = g.T
        self.g_lb = g_lb.T
        self.g_ub = g_ub.T

        # Generate solver
        self.x_nlp = ca.vertcat(
            ca.reshape(z, (self.N + 1) * self.n, 1),
            ca.reshape(v, self.N * self.m, 1),
        )
        self.p_nlp = ca.vertcat(
            p_z0.T,
            ca.reshape(p_z_ref, (self.N + 1) * self.n, 1), ca.reshape(p_v_ref, self.N * self.m, 1),
        )

        # x_cols, g_cols, p_cols = generate_col_names(N, x_nlp, g, p_nlp)
        nlp_dict = {
            "x": self.x_nlp,
            "f": self.obj,
            "g": self.g,
            "p": self.p_nlp
        }
        self.nlp_opts = {
            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-4,
            # "ipopt.print_level": 5,
            "print_time": True,
        }

        self.nlp_solver = ca.nlpsol("DeepTubeMPC", "ipopt", nlp_dict, self.nlp_opts)