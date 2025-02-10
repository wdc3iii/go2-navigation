import numpy as np
from scipy.interpolate import interp1d


def track_trajectory(z_path, v_path, t_path, t, z, kp, v_min, v_max):
    interpolator = interp1d(t_path, z_path, axis=0, kind='linear', fill_value='extrapolate')
    pos_d = interpolator(t)
    v_d = v_path[np.searchsorted(t_path, t) - 1, :]
    
    pos_err = z[:2] - pos_d[:2]
    heading_err = z[2] - pos_d[2]

    # convert position error into local frame
    pos_err = np.array([
        np.cos(z[2]) * pos_err[0] + np.sin(z[2]) * pos_err[1],
        -np.sin(z[2]) * pos_err[0] + np.cos(z[2]) * pos_err[1]
    ])
    # Put feedforward velocity into local frame (heading error, since v_ff defined in local frame ON TRAJECTORY)
    v_ff = np.array([
        np.cos(heading_err) * v_d[0] + np.sin(heading_err) * v_d[1],
        -np.sin(heading_err) * v_d[1] + np.cos(heading_err) * v_d[1],
        v_d[2]
    ])
    v_cmd = np.clip(
        v_ff - kp * np.hstack((pos_err, heading_err)),
        v_min,
        v_max
    )
    return v_cmd
