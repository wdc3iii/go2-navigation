import numpy as np
import casadi as ca
from scipy.spatial import KDTree
from go2_dyn_tube_mpc.map_utils import MapUtils
from scipy.interpolate import interp1d

class DynamicTubeMPC:
    
    def __init__(self, dt, N, n, m, Q, Qf, R, Rv_f, Rv_s, Q_sched, v_min, v_max,
            robot_radius=0, map_resolution=0.05, obs_rho=10, obs_constraint_method="QuadraticPenalty", fix_internal_constraints=False):
        # Dimensions
        self.N = N
        self.n = n
        self.m = m
        self.dt = dt

        # Cost function parameters
        self.Q = np.diag(Q[:2])
        self.Q_heading = Q[-1]
        self.Qf = np.diag(Qf[:2])
        self.Qf_heading = Qf[-1]
        self.R = np.diag(R)
        self.Rv_f = np.diag(Rv_f)
        self.Rv_s = np.diag(Rv_s)
        self.Q_sched = Q_sched

        # Input bounds
        self.v_max_bound = v_max
        self.v_max = np.zeros((m,))
        self.v_min_bound = v_min
        self.v_min = np.zeros((m,))

        # Reference Trajectory
        self.path = None
        self.path_length = None
        self.map_resolution = map_resolution
        self.ref_length = np.arange(self.N + 1) * self.v_max_bound[0] * dt
        self.z_ref = np.zeros((N + 1, n))
        self.v_ref = np.zeros((N, m))
        self.v_ref[:, 0] = self.v_max_bound[0]

        # Initial condition
        self.z0 = np.zeros((n,))

        # Warm starts
        self.z_warm = np.zeros((N + 1, n))
        self.v_warm = np.zeros((N, m))

        # Constraint parameters
        self.obs_rho = obs_rho
        self.obs_constraint_method = obs_constraint_method
        self.robot_radius = robot_radius
        assert self.obs_constraint_method in ["Constraint", "QuadraticPenalty", "LinearPenalty"]

        # Map / scan
        self.map_nearest_inds = None
        self.map_unc_occ_nearest_inds = None
        self.scan = None
        self.fix_internal_constraints = fix_internal_constraints

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
        z, v = self.extract_solution(sol)
        self.z_warm = z.copy()
        self.v_warm = v.copy()
        return z, v

    def set_input_bounds(self, v_min, v_max):
        assert v_min.shape == self.v_min.shape and v_max.shape == self.v_max.shape
        self.v_min = np.clip(v_min, self.v_min_bound, np.zeros((3,)))
        self.v_max = np.clip(v_max, np.zeros((3,)), self.v_max_bound)

    def set_path(self, path):
        self.path = self.map.map_to_pose(path)
        distances = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.path_length = np.insert(np.cumsum(distances), 0, 0)

    def set_initial_condition(self, z0):
        assert z0.shape == self.z0.shape
        self.z0 = z0

    def set_warm_start(self, z_warm, v_warm):
        assert z_warm.shape == self.z_warm.shape and v_warm.shape == self.v_warm.shape
        self.z_warm = z_warm
        self.v_warm = v_warm

    def reset_warm_start(self):
        self.compute_reference()
        self.z_warm = self.z_ref.copy()
        self.v_warm = self.v_ref.copy()

    def update_scan(self, scan_points):
        self.scan = scan_points

    def update_nearest_inds(self, nearest_inds):
        # TODO: update nearest_inds
        pass

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
        # Compute constraints
        A, b = self.compute_constraints()
        # Compute reference
        self.compute_reference()

        # Assemble params
        params = np.vstack([
            self.z0[:, None],
            np.reshape(self.z_ref, (-1, 1), order='F'),
            np.reshape(self.v_ref, (-1, 1), order='F'),
            np.reshape(A, (-1, 1), order='F'),
            b[:, None]
        ])
        return params

    def compute_constraints(self):
        # Constraint buffer, size of voxel + robot radius
        constraint_buffer = -np.sqrt(2) * self.map_resolution / 2 - self.robot_radius

        # Combine map and scan
        use_scan = self.nearest_scan_dists < np.abs(self.nearest_map_dists)
        nearest_points = np.where(use_scan[:, None], self.nearest_scan_points, self.nearest_map_points)
        # TODO: Filter normals with adjacent occupancy information?
        normals = self.z_warm[:, :2] - nearest_points

        # Compute normals
        A = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        b = -np.sum(A * nearest_points, axis=1) + constraint_buffer

        if self.fix_internal_constraints:
            A[self.nearest_map_dists < 0] *= -1
            b[self.nearest_map_dists] += (np.sqrt(2) + 1) * self.map_resolution / 2
        return A, b

    @staticmethod
    def smooth_heading(v):
        pad = 5  # odd
        kernel = np.ones(pad) / pad
        return np.convolve(np.pad(v, pad_width=pad//2, mode='edge'), kernel, 'valid')

    def compute_reference(self):
        # Compute reference
        init_ind = np.argmin(np.linalg.norm(self.path - self.z0[:2], axis=1))
        path_length = self.path_length[init_ind:] - self.path_length[init_ind]
        interp_x = interp1d(path_length, self.path[init_ind:, 0],
            kind='linear', fill_value=(self.path[0, 0], self.path[-1, 0]),
            bounds_error=False, assume_sorted=True)(self.ref_length)
        interp_y = interp1d(path_length, self.path[init_ind:, 1],
            kind='linear', fill_value=(self.path[0, 1], self.path[-1, 1]),
            bounds_error=False, assume_sorted=True)(self.ref_length)
        diff_y = np.diff(interp_y)
        diff_x = np.diff(interp_x)
        headings = np.arctan2(diff_y, diff_x)

        self.z_ref = np.hstack([interp_x[:, None], interp_y[:, None], np.insert(headings, -1, headings[-1])[:, None]])
        self.z_ref[:, -1] = self.smooth_heading(self.z_ref[:, -1])
        self.v_ref[:, -1] = np.clip(np.diff(self.z_ref[:, -1]) / self.dt, self.v_min_bound[-1], self.v_max_bound[-1])
        self.v_ref[np.logical_and(diff_x == 0, diff_y == 0)] = 0

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

    @staticmethod
    def geo_cost(x, Q, goal=None, t_sched=None):
        if goal is None:
            dist = x
        else:
            if goal.shape == x.shape:
                dist = x - goal
            else:
                dist = x - ca.repmat(goal, x.shape[0], 1)

        obj = Q * (ca.sin(dist) * ca.sin(dist) + (1 - ca.cos(dist)) * (1 - ca.cos(dist)))
        if t_sched is not None:
            obj *= t_sched
        return ca.sum1(obj)

    def lateral_unicycle_dynamics(self, z, v):
        gv_0 = v[0] * ca.cos(z[2]) - v[1] * ca.sin(z[2])
        gv_1 = v[0] * ca.sin(z[2]) + v[1] * ca.cos(z[2])
        gv_2 = v[2]
        return z + ca.vertcat(gv_0, gv_1, gv_2) * self.dt

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

    def get_nearest_map_points(self):
        # TODO: need occupancy grid of warm start...
        warm_x, warm_y = self.map.pose_to_map(self.z_warm)

        nearest_points = self.map_nearest_inds[:, warm_x, warm_y].T
        nearest_points = self.map.map_to_pose(nearest_points)
        dists = np.linalg.norm(nearest_points - self.z_warm[:, :2], axis=1)
        return nearest_points, dists

    def get_nearest_occ_map_points(self, invalid_inds):
        warm_x, warm_y = self.map.pose_to_map(self.z_warm[invalid_inds])

        nearest_points = self.map_unc_occ_nearest_inds[:, warm_x, warm_y].T
        nearest_points = self.map.map_to_pose(nearest_points)
        dists = np.linalg.norm(nearest_points - self.z_warm[invalid_inds, :2], axis=1)
        return nearest_points, dists

    def get_nearest_scan_points(self):
        # Computes a KDTree of the scan points, and queries for closest distances
        # O(NlogM) < O(NM) - beats brute force
        tree = KDTree(self.scan)
        dists, inds = tree.query(self.z_warm[:, :2])  # Find nearest neighbor indices
        return self.scan[inds], dists

    def setup_ocp(self):
        self.z_lb = ca.DM(self.N + 1, self.n)
        self.z_lb[:] = -ca.inf
        self.z_ub = ca.DM(self.N + 1, self.n)
        self.z_ub[:] = ca.inf

        # State variables
        p = ca.MX.sym("p", self.N + 1, self.n - 1)
        theta = ca.MX.sym("theta", self.N + 1, 1)
        z = ca.horzcat(p, theta)
        v = ca.MX.sym("v", self.N, self.m)

        # Parameters for z_ref
        p_z_ref = ca.MX.sym("p_z_ref", self.N + 1, self.n)
        p_v_ref = ca.MX.sym("p_v_ref", self.N, self.m)

        # Obstacle normals
        p_A = ca.MX.sym("p_A", self.N + 1, self.n - 1)
        p_b = ca.MX.sym("p_b", self.N + 1, 1)

        # Initial condition parameter
        p_z0 = ca.MX.sym("p_z0", 1, self.n)  # Initial projection Pz(x0) state

        # Define cost function
        # Reference Tracking
        self.obj = self.quadratic_objective(z[:-1, :2], self.Q, goal=p_z_ref[:-1, :2], t_sched=self.Q_sched) \
              + self.geo_cost(z[:-1, 2], self.Q_heading, goal=p_z_ref[:-1, 2], t_sched=self.Q_sched) \
              + self.quadratic_objective(v, self.R, goal=p_v_ref) \
              + self.quadratic_objective(z[-1, :2], self.Qf, goal=p_z_ref[-1, :2]) \
              + self.geo_cost(z[-1, 2], self.Qf_heading, goal=p_z_ref[-1, 2])

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
        # Initial Condition Constraint
        g_ic, g_lb_ic, g_ub_ic = self.equality_constraint(z[0, :], p_z0)

        # TODO: define tube constraint
        # Linearized Obstacle Constraint A p + b >= 0
        obs_dist = ca.sum2(p_A * p) + p_b
        if self.obs_constraint_method == "Constraint":    # Constraint method
            g_obs = obs_dist
            g_lb_obs = ca.DM(self.N + 1, 1)
            g_ub_obs = ca.DM(self.N + 1, 1)
            g_ub_obs[:] = ca.inf

            # Create constraints
            g = ca.horzcat(g_dyn, g_ic, g_obs.T)
            g_lb = ca.horzcat(g_dyn_lb, g_lb_ic, g_lb_obs.T)
            g_ub = ca.horzcat(g_dyn_ub, g_ub_ic, g_ub_obs.T)
        else:
            g = ca.horzcat(g_dyn, g_ic)
            g_lb = ca.horzcat(g_dyn_lb, g_lb_ic)
            g_ub = ca.horzcat(g_dyn_ub, g_ub_ic)
            if self.obs_constraint_method == "QuadraticPenalty":  # quadratic cost method
                self.obj += self.obs_rho / 2 * ca.sumsqr(ca.fmin(obs_dist, 0))
            elif self.obs_constraint_method == "LinearPenalty":        # linear cost method
                self.obj += self.obs_rho * ca.sum1(ca.fmin(obs_dist, 0))
            else:
                raise RuntimeError("Unknown obs_constraint_method")

        # Constraint matrices
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
            ca.reshape(p_A, (self.N + 1) * (self.n - 1), 1), p_b
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