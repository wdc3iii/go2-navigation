from go2_dyn_tube_mpc.map_utils import map_to_pose, pose_to_map
from go2_dyn_tube_mpc.load_tube_model import load_tube_model

import torch
import numpy as np
import casadi as ca
import l4casadi as l4c
from scipy.io import savemat
from scipy.spatial import KDTree
from scipy.interpolate import interp1d


class DynamicTubeMPC:
    
    def __init__(self, dt, N, H, n, m, Q, Qf, R, Rv_f, Rv_s, Q_sched, v_min, v_max, model_path, device,
                 robot_radius=0, map_resolution=0.05, obs_rho=100, scan_rel_dist_cap=0.5, normal_alpha=0.5,
                 w_max=2, obs_constraint_method="Constraint", fix_internal_constraints=False,
                 solver_str="snopt", verbosity=0):
        # Dimensions
        self.N = N
        self.H = H
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
        self.w_max = w_max

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
        self.e = np.zeros((H,))
        self.v_prev = np.zeros((H, m))

        # Warm starts
        self.z_warm = np.zeros((N + 1, n))
        self.v_warm = np.zeros((N, m))
        self.w_warm = np.zeros((N, n))

        # Constraint parameters
        self.obs_rho = obs_rho
        self.obs_constraint_method = obs_constraint_method
        self.robot_radius = robot_radius
        self.scan_rel_dist_cap = scan_rel_dist_cap
        self.normal_alpha = normal_alpha
        assert self.obs_constraint_method in ["Constraint", "QuadraticPenalty", "LinearPenalty"]
        self.model_path = model_path
        self.device = device
        self.solver_str = solver_str
        self.verbosity = verbosity

        # Map / scan
        self.map_nearest_inds = None
        self.map_nearest_dists = None
        self.map_origin = np.zeros((2,))
        self.map_theta = 0
        self.map_unc_occ_nearest_inds = None
        self.scan = None
        self.fix_internal_constraints = fix_internal_constraints

        # Casadi variables
        self.z_lb = None
        self.z_ub = None
        self.w_lb = None
        self.w_ub = None
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
            x0=x_init, p=params, lbg=self.g_lb, ubg=self.g_ub, lbx=self.lbx(), ubx=self.ubx()
        )
        z, v, w = self.extract_solution(sol)
        stats = self.nlp_solver.stats()

        # # TODO: Debugging
        # with open("debugging_dtmpc.csv", 'w') as f:
        #     data = [
        #         ['Q', *self.Q.flatten().tolist()],
        #         ['Qf', *self.Qf.flatten().tolist()],
        #         ['Q_heading', self.Q_heading],
        #         ['Qf_heading', self.Qf_heading],
        #         ['R', *self.Q.flatten().flatten().tolist()],
        #         ['Rv_f', *self.Rv_f.flatten().tolist()],
        #         ['Rv_s', *self.Rv_s.flatten().tolist()],
        #         ['Q_sched', *self.Q_sched.flatten().tolist()],
        #         ['vmin', *self.v_min.flatten().tolist()],
        #         ['vmax', *self.v_max.flatten().tolist()],
        #         ['zref', *self.z_ref.flatten().tolist()],
        #         ['vref', *self.v_ref.flatten().tolist()],
        #         ['z0', *self.z0.flatten().tolist()],
        #         ['z_warm', *self.z_warm.flatten().tolist()],
        #         ['v_warm', *self.v_warm.flatten().tolist()],
        #         ['z', *z.flatten().tolist()],
        #         ['v', *v.flatten().tolist()]
        #     ]
        #     import csv
        #     writer = csv.writer(f)
        #     writer.writerows(data)
        savemat("sol.mat", {
            "z": z,
            "v": v,
            "zwarm": self.z_warm
        })
        self.z_warm = z.copy()
        self.v_warm = v.copy()
        self.w_warm = w.copy()

        return z, v, w, stats["success"]

    def set_input_bounds(self, v_min, v_max):
        assert v_min.shape == self.v_min.shape and v_max.shape == self.v_max.shape
        self.v_min = np.clip(v_min, self.v_min_bound, np.zeros((3,)))
        self.v_max = np.clip(v_max, np.zeros((3,)), self.v_max_bound)

    def set_path(self, path):
        self.path = path
        distances = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.path_length = np.insert(np.cumsum(distances), 0, 0)

    def set_initial_condition(self, z0):
        assert z0.shape == self.z0.shape
        self.z0 = z0

    def set_warm_start(self, z_warm, v_warm, w_warm=None):
        assert z_warm.shape == self.z_warm.shape and v_warm.shape == self.v_warm.shape
        self.z_warm = z_warm
        self.v_warm = v_warm
        if w_warm is None:
            self.w_warm = np.ones((self.N,)) * 0.2
        else:
            self.w_warm = w_warm

    def reset_warm_start(self):
        self.compute_reference()
        self.z_warm = self.z_ref.copy()
        self.v_warm = self.v_ref.copy()
        self.w_warm = np.ones((self.N,)) * 0.2

    def update_history(self, e0, v0):
        self.e[1:] = self.e[:-1]
        self.e[0] = e0
        self.v_prev[1:] = self.v_prev[:-1]
        self.v_prev[0] = v0

    def update_scan(self, scan_points):
        self.scan = scan_points

    def update_nearest_inds(self, map_nearest_inds, nearest_map_dists, map_origin, map_theta):
        self.map_nearest_inds = map_nearest_inds
        self.map_nearest_dists = nearest_map_dists
        self.map_origin = map_origin
        self.map_theta = map_theta

    def lbx(self):
        v_lb = ca.DM(np.repeat(self.v_min[:, None], self.N, 0))
        return ca.vertcat(
            ca.reshape(self.z_lb, (self.N + 1) * self.n, 1),
            ca.reshape(v_lb, self.N * self.m, 1),
            ca.DM(self.N, 1)
        )

    def ubx(self):
        v_ub = ca.DM(np.repeat(self.v_max[:, None], self.N, 0))
        return ca.vertcat(
            ca.reshape(self.z_ub, (self.N + 1) * self.n, 1),
            ca.reshape(v_ub, self.N * self.m, 1),
            ca.DM(np.ones((self.N, 1)) * self.w_max)
        )

    def init_params(self):
        A, b, _ = self.compute_constraints()  # Compute constraints
        self.compute_reference()              # Compute reference

        # Assemble params
        params = np.vstack([
            self.z0[:, None],
            np.reshape(self.z_ref, (-1, 1), order='F'),
            np.reshape(self.v_ref, (-1, 1), order='F'),
            np.reshape(A, (-1, 1), order='F'),
            b[:, None],
            self.v_min[:, None], self.v_max[:, None],
            self.e[:, None], np.reshape(self.v_prev, (-1, 1), order='F')
        ])
        return params

    def compute_constraints(self):
        # Constraint buffer, size of voxel + robot radius
        constraint_buffer = -np.sqrt(2) * self.map_resolution / 2 - self.robot_radius
        nearest_map_normals, nearest_map_dists, nearest_map_points = self.get_nearest_map_normals()
        nearest_scan_normals, nearest_scan_dists, nearest_scan_points = self.get_nearest_scan_normals()

        # Combine map and scan
        use_scan = nearest_scan_dists < np.abs(nearest_map_dists)
        normals = np.where(use_scan[:, None], nearest_scan_normals, nearest_map_normals)
        nearest_points = np.where(use_scan[:, None], nearest_scan_points, nearest_map_points)

        # Compute normals
        A = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        b = -np.sum(A * nearest_points, axis=1) + constraint_buffer

        # if self.fix_internal_constraints:
        #     # A[nearest_map_dists < 0] *= -1
        #     b[nearest_map_dists < 0] += (np.sqrt(2) + 1) * self.map_resolution / 2

        # Debugging
        savemat("constraints.mat", {
            "A": A, "b": b,
            "nearest_map_points": nearest_map_points,
            "nearest_map_dists": nearest_map_dists,
            "nearest_scan_points": nearest_scan_points,
            "nearest_scan_dists": nearest_scan_dists,
            "nearest_points": nearest_points,
            "zwarm": self.z_warm
        })
        return A, b, nearest_points

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
        a_star_finished = np.logical_and(diff_x == 0, diff_y == 0)
        if np.any(a_star_finished):
            self.v_ref[a_star_finished] = 0
            a_star_finished = np.insert(a_star_finished, 0, 0)
            self.z_ref[a_star_finished, 2] = self.z_ref[np.argmax(a_star_finished) - 1, 2]

    def init_decision_var(self):
        x_init = np.vstack([
            np.reshape(self.z_warm, ((self.N + 1) * self.n, 1), order='F'),
            np.reshape(self.v_warm, (self.N * self.m, 1), order='F'),
            self.w_warm[:, None]
        ])
        return x_init

    def extract_solution(self, sol):
        z_ind = (self.N + 1) * self.n
        v_ind = self.N * self.m
        z_sol = np.array(sol["x"][:z_ind, :].reshape((self.N + 1, self.n)))
        v_sol = np.array(sol["x"][z_ind:z_ind + v_ind, :].reshape((self.N, self.m)))
        w_sol = np.array(sol["x"][z_ind + v_ind:, :]).reshape((self.N,))
        return z_sol, v_sol, w_sol

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
        warm_x, warm_y = pose_to_map(self.z_warm[1:], self.map_origin, self.map_theta, self.map_resolution)
        warm_x = np.clip(warm_x, 0, self.map_nearest_dists.shape[0] - 1)
        warm_y = np.clip(warm_y, 0, self.map_nearest_dists.shape[1] - 1)
        nearest_dists = self.map_nearest_dists[warm_x, warm_y].T
        nearest_inds = self.map_nearest_inds[:, warm_x, warm_y].T
        nearest_points = map_to_pose(nearest_inds, self.map_origin, self.map_theta, self.map_resolution)
        dists = np.linalg.norm(nearest_points - self.z_warm[1:, :2], axis=1)
        dists[nearest_dists < 0] *= -1
        return nearest_points, dists, nearest_inds
    
    def get_nearest_map_normals(self):
        # Get the nearest points on the map
        nearest_points, dists, nearest_inds = self.get_nearest_map_points()

        # Compute normal from the gradients
        grad_normals = np.zeros_like(nearest_points)
        for i in range(nearest_points.shape[0]):
            nearest_ind = nearest_inds[i]
            grad_normals[i, :] = self.compute_map_obstacle_normal(nearest_ind)

        # Compute normals from the points
        proj_normals = self.z_warm[1:, :2] - nearest_points
        proj_normals /= np.linalg.norm(proj_normals, axis=-1, keepdims=True)
        proj_normals[dists < 0] *= -1

        # Debugging
        # savemat("constraints_construction.mat", {
        #     "A_proj": proj_normals, "b_proj": -np.sum(proj_normals * nearest_points, axis=1),
        #     "A_grad": grad_normals, "b_grad": -np.sum(grad_normals * nearest_points, axis=1),
        #     "nearest_points": nearest_points,
        #     "nearest_dists": dists,
        #     "zwarm": self.z_warm
        # })
        # Intelligently mix normal representations
        normals = self.normal_alpha * proj_normals + (1 - self.normal_alpha) * grad_normals
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals, dists, nearest_points
    
    def compute_map_obstacle_normal(self, point):
        x = point[0]
        y = point[1]
        xp1 = min(x + 1, self.map_nearest_dists.shape[0] - 1)
        xm1 = max(x - 1, 0)
        yp1 = min(y + 1, self.map_nearest_dists.shape[1] - 1)
        ym1 = max(y - 1, 0)
        dx = (
            4 * (self.map_nearest_dists[xp1, y] - self.map_nearest_dists[xm1, y])
            + (self.map_nearest_dists[xp1, ym1] - self.map_nearest_dists[xm1, ym1])
            + (self.map_nearest_dists[xp1, yp1] - self.map_nearest_dists[xm1, yp1])
        ) / 8.0
        
        dy = (
            4 * (self.map_nearest_dists[x, yp1] - self.map_nearest_dists[x, ym1])
            + (self.map_nearest_dists[xp1, yp1] - self.map_nearest_dists[xp1, ym1])
            + (self.map_nearest_dists[xm1, yp1] - self.map_nearest_dists[xm1, ym1])
        ) / 8.0
    
        # Normalize the gradient to get the unit normal
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        return np.array((dx / norm, dy / norm))  # safety function positive outside obstacles, gradient points out

    def get_nearest_scan_points(self):
        # Computes a KDTree of the scan points, and queries for closest distances
        # O(NlogM) < O(NM) - beats brute force
        tree = KDTree(self.scan)
        dists, inds = tree.query(self.z_warm[1:, :2])  # Find nearest neighbor indices
        return self.scan[inds], dists, inds
    
    def get_nearest_scan_normals(self):
        nearest_points, dists, nearest_inds = self.get_nearest_scan_points()
        normals = np.zeros_like(nearest_points)
        for i in range(nearest_points.shape[0]):
            nearest_ind = nearest_inds[i]
            normals[i, :] = self.compute_scan_obstacle_normal(nearest_ind)

        savemat("scan_constraints_construction.mat", {
            "A": normals, "b": -np.sum(normals * nearest_points, axis=1),
            "nearest_points": nearest_points,
            "nearest_dists": dists,
            "zwarm": self.z_warm
        })
        return normals, dists, nearest_points
    
    def compute_scan_obstacle_normal(self, point):
        pts = [(point - 1) % self.scan.shape[0], point, (point + 1) % self.scan.shape[0]]
        neighbors = self.scan[pts]
        rel_dist = np.linalg.norm(np.diff(neighbors, axis=0), axis=-1)

        if rel_dist[0] > self.scan_rel_dist_cap and rel_dist[1] > self.scan_rel_dist_cap:
            normal = np.zeros((2,))  # Suprious point
        else:
            # if rel_dist[0] > self.scan_rel_dist_cap:
            #     x = neighbors[1:, 0]
            #     y = neighbors[1:, 1]
            # elif rel_dist[1] > self.scan_rel_dist_cap:
            #     x = neighbors[:-1, 0]
            #     y = neighbors[:-1, 1]
            # else:
            x = neighbors[:, 0]
            y = neighbors[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T  # Linear regression matrix
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            norm_length = np.sqrt(1 + slope**2) + 1e-6
            normal = np.array([-slope, 1]) / norm_length  # Perpendicular to the tangent
        # Ensure the normal points inward
        to_center = self.z0[:2] - self.scan[point]
        if np.dot(normal, to_center) < 0:
            normal = -normal  # Flip the normal to point inward
        return normal

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
        w = ca.MX.sym("w", self.N, 1)
        self.w_lb = ca.DM(self.N, 1)
        self.w_ub = ca.DM(np.ones((self.N, 1)) * self.w_max)

        # Parameters for z_ref
        p_z_ref = ca.MX.sym("p_z_ref", self.N + 1, self.n)
        p_v_ref = ca.MX.sym("p_v_ref", self.N, self.m)
        p_v_min = ca.MX.sym("p_v_min", 1, self.m)
        p_v_max = ca.MX.sym("p_v_max", 1, self.m)

        # Obstacle normals
        p_A = ca.MX.sym("p_A", self.N, self.n - 1)
        p_b = ca.MX.sym("p_b", self.N, 1)

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
        recursive_nn_tube_dyn, H_fwd, H_rev, eval_nn_tube_dyn = self.get_recursive_nn_tube_dynamics()
        e = ca.MX.sym("e", H_rev, 1)
        v_prev = ca.MX.sym("v_prev", H_rev, self.m)
        g_tube, g_lb_tube, g_ub_tube = recursive_nn_tube_dyn(z, v, w, e, v_prev)

        # Linearized Obstacle Constraint A p + b - w >= 0
        obs_dist = ca.sum2(p_A * p[1:, :]) + p_b - w  # Constraint applys only after the current node (which can't be moved)
        if self.obs_constraint_method == "Constraint":    # Constraint method
            g_obs = obs_dist
            g_lb_obs = ca.DM(self.N, 1)
            g_ub_obs = ca.DM(self.N, 1)
            g_ub_obs[:] = ca.inf

            # Create constraints
            g = ca.horzcat(g_dyn, g_ic, g_tube, g_obs.T)
            g_lb = ca.horzcat(g_dyn_lb, g_lb_ic, g_lb_tube, g_lb_obs.T)
            g_ub = ca.horzcat(g_dyn_ub, g_ub_ic, g_ub_tube, g_ub_obs.T)
        else:
            g = ca.horzcat(g_dyn, g_ic, g_tube)
            g_lb = ca.horzcat(g_dyn_lb, g_lb_ic, g_lb_tube)
            g_ub = ca.horzcat(g_dyn_ub, g_ub_ic, g_ub_tube)
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
            w
        )
        self.p_nlp = ca.vertcat(
            p_z0.T,
            ca.reshape(p_z_ref, (self.N + 1) * self.n, 1), ca.reshape(p_v_ref, self.N * self.m, 1),
            ca.reshape(p_A, self.N * (self.n - 1), 1), p_b,
            p_v_min.T, p_v_max.T,
            e, ca.reshape(v_prev, -1, 1)
        )

        # x_cols, g_cols, p_cols = generate_col_names(N, x_nlp, g, p_nlp)
        nlp_dict = {
            "x": self.x_nlp,
            "f": self.obj,
            "g": self.g,
            "p": self.p_nlp
        }

        if self.solver_str == "ipopt":
            self.nlp_opts = {
                "ipopt.linear_solver": "mumps",
                "ipopt.sb": "yes",
                "ipopt.max_iter": 200,
                "ipopt.tol": 1e-4,
                "print_time": False,
                "ipopt.print_level": self.verbosity,  # Further suppress IPOPT messages
            }
        elif self.solver_str == "snopt":
            self.nlp_opts = {"snopt": {
                "Major feasibility tolerance": 1e-6,
                "Major optimality tolerance": 1e-3,
                "Major iterations limit": 4,
                "Minor iterations limit": 20,
                "Time limit": 0.1
            }}
        else:
            raise RuntimeError("Unknown solver_str")

        self.nlp_solver = ca.nlpsol("DeepTubeMPC", self.solver_str, nlp_dict, self.nlp_opts)

    def get_recursive_nn_tube_dynamics(self):
        tube_recursive_model = load_tube_model(self.model_path)

        tube_recursive_model.mlp.to(self.device)
        tube_recursive_model.mlp.eval()
        fw = l4c.L4CasADi(tube_recursive_model.mlp, device=self.device)

        def recursive_nn_tube_dyn(z, v, w, e, v_prev):
            v = ca.vertcat(v_prev, v)
            all_data = []
            for i in range(w.shape[0]):
                if i < tube_recursive_model.H_rev:
                    data = ca.horzcat(
                        e[i:, :].T, w[0:i, :].T,
                        ca.reshape(v[i:i + tube_recursive_model.H_rev, :].T, 1,
                                   tube_recursive_model.H_rev * v.shape[1]),
                        ca.DM([i])
                    )
                else:
                    data = ca.horzcat(
                        w[i - tube_recursive_model.H_rev:i, :].T,
                        ca.reshape(v[i:i + tube_recursive_model.H_rev, :].T, 1,
                                   tube_recursive_model.H_rev * v.shape[1]),
                        ca.DM([i])
                    )
                all_data.append(data)

            g = ca.horzcat(*[fw(data) for data in all_data]) - w.T
            g_lb = ca.DM(*g.shape)
            g_ub = ca.DM(*g.shape)

            return g, g_lb, g_ub

        def eval_nn_tube_dyn(z, v, e, v_prev):
            w = torch.zeros((v.shape[0], 1), device=self.device)
            v = torch.from_numpy(v).float().to(self.device)
            e = torch.from_numpy(e).float().to(self.device)
            v_prev = torch.from_numpy(v_prev).float().to(self.device)
            v = torch.concatenate((v_prev, v))
            for i in range(w.shape[0]):
                if i < tube_recursive_model.H_rev:
                    data = torch.concatenate((
                        e[i:, :].T, w[0:i, :].T,
                        torch.reshape(v[i:i + tube_recursive_model.H_rev, :],
                                      (1, tube_recursive_model.H_rev * v.shape[1])),
                        torch.tensor([[i]], device=self.device).float()
                    ), dim=1)
                else:
                    data = torch.concatenate((
                        w[i - tube_recursive_model.H_rev:i, :].T,
                        torch.reshape(v[i:i + tube_recursive_model.H_rev, :],
                                      (1, tube_recursive_model.H_rev * v.shape[1])),
                        torch.tensor([[i]], device=self.device)
                    ), dim=1)
                w[i] = tube_recursive_model.mlp(data)

            return w

        return recursive_nn_tube_dyn, tube_recursive_model.H_fwd, tube_recursive_model.H_rev, eval_nn_tube_dyn

    # def setup_ocp(self):
    #     self.z_lb = ca.DM(self.N + 1, self.n)
    #     self.z_lb[:] = -ca.inf
    #     self.z_ub = ca.DM(self.N + 1, self.n)
    #     self.z_ub[:] = ca.inf
    #
    #     # State variables
    #     p = ca.MX.sym("p", self.N + 1, self.n - 1)
    #     theta = ca.MX.sym("theta", self.N + 1, 1)
    #     z = ca.horzcat(p, theta)
    #     v = ca.MX.sym("v", self.N, self.m)
    #
    #     # Parameters for z_ref
    #     p_z_ref = ca.MX.sym("p_z_ref", self.N + 1, self.n)
    #     p_v_ref = ca.MX.sym("p_v_ref", self.N, self.m)
    #
    #     # Obstacle normals
    #     p_A = ca.MX.sym("p_A", self.N + 1, self.n - 1)
    #     p_b = ca.MX.sym("p_b", self.N + 1, 1)
    #
    #     # Initial condition parameter
    #     p_z0 = ca.MX.sym("p_z0", 1, self.n)  # Initial projection Pz(x0) state
    #
    #     # Define cost function
    #     # Reference Tracking
    #     self.obj = self.quadratic_objective(z[:-1, :2], self.Q, goal=p_z_ref[:-1, :2], t_sched=self.Q_sched) \
    #           + self.geo_cost(z[:-1, 2], self.Q_heading, goal=p_z_ref[:-1, 2], t_sched=self.Q_sched) \
    #           + self.quadratic_objective(v, self.R, goal=p_v_ref) \
    #           + self.quadratic_objective(z[-1, :2], self.Qf, goal=p_z_ref[-1, :2]) \
    #           + self.geo_cost(z[-1, 2], self.Qf_heading, goal=p_z_ref[-1, 2])
    #
    #     # Smoothness of input
    #     # First order difference
    #     if np.any(self.Rv_f > 0):
    #         Rv_f_ = ca.DM(self.Rv_f)
    #         self.obj += self.quadratic_objective(v[:-1, :] - v[1:, :], Rv_f_)
    #     # Second order difference
    #     if np.any(self.Rv_s > 0):
    #         Rv_s_ = ca.DM(self.Rv_s)
    #         first = v[:-1, :] - v[1:, :]
    #         self.obj += self.quadratic_objective(first[:-1, :] - first[1:, :], Rv_s_)
    #
    #     # Define dynamics constraint
    #     g_dyn, g_dyn_lb, g_dyn_ub = self.dynamics_constraints(z, v)
    #     # Initial Condition Constraint
    #     g_ic, g_lb_ic, g_ub_ic = self.equality_constraint(z[0, :], p_z0)
    #
    #     # TODO: define tube constraint
    #     # Linearized Obstacle Constraint A p + b >= 0
    #     obs_dist = ca.sum2(p_A * p) + p_b
    #     if self.obs_constraint_method == "Constraint":    # Constraint method
    #         g_obs = obs_dist
    #         g_lb_obs = ca.DM(self.N + 1, 1)
    #         g_ub_obs = ca.DM(self.N + 1, 1)
    #         g_ub_obs[:] = ca.inf
    #
    #         # Create constraints
    #         g = ca.horzcat(g_dyn, g_ic, g_obs.T)
    #         g_lb = ca.horzcat(g_dyn_lb, g_lb_ic, g_lb_obs.T)
    #         g_ub = ca.horzcat(g_dyn_ub, g_ub_ic, g_ub_obs.T)
    #     else:
    #         g = ca.horzcat(g_dyn, g_ic)
    #         g_lb = ca.horzcat(g_dyn_lb, g_lb_ic)
    #         g_ub = ca.horzcat(g_dyn_ub, g_ub_ic)
    #         if self.obs_constraint_method == "QuadraticPenalty":  # quadratic cost method
    #             self.obj += self.obs_rho / 2 * ca.sumsqr(ca.fmin(obs_dist, 0))
    #         elif self.obs_constraint_method == "LinearPenalty":        # linear cost method
    #             self.obj += self.obs_rho * ca.sum1(ca.fmin(obs_dist, 0))
    #         else:
    #             raise RuntimeError("Unknown obs_constraint_method")
    #
    #     # Constraint matrices
    #     self.g = g.T
    #     self.g_lb = g_lb.T
    #     self.g_ub = g_ub.T
    #
    #     # Generate solver
    #     self.x_nlp = ca.vertcat(
    #         ca.reshape(z, (self.N + 1) * self.n, 1),
    #         ca.reshape(v, self.N * self.m, 1),
    #     )
    #     self.p_nlp = ca.vertcat(
    #         p_z0.T,
    #         ca.reshape(p_z_ref, (self.N + 1) * self.n, 1), ca.reshape(p_v_ref, self.N * self.m, 1),
    #         ca.reshape(p_A, (self.N + 1) * (self.n - 1), 1), p_b
    #     )
    #
    #     # x_cols, g_cols, p_cols = generate_col_names(N, x_nlp, g, p_nlp)
    #     nlp_dict = {
    #         "x": self.x_nlp,
    #         "f": self.obj,
    #         "g": self.g,
    #         "p": self.p_nlp
    #     }
    #     self.nlp_opts = {
    #         "ipopt.linear_solver": "mumps",
    #         "ipopt.sb": "yes",
    #         "ipopt.max_iter": 200,
    #         "ipopt.tol": 1e-4,
    #         "print_time": False,
    #         "ipopt.print_level": 0,  # Further suppress IPOPT messages
    #         # "ipopt.sb": "yes",  # Suppresses IPOPT banner
    #     }
    #
    #     self.nlp_solver = ca.nlpsol("DeepTubeMPC", "ipopt", nlp_dict, self.nlp_opts)
