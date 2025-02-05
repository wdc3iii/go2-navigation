# from casadi import *
#
# N_refpath = 10
# ref_path_s = np.linspace(0, 1, N_refpath)
# p = MX.sym('p', N_refpath, 1)
# x = MX.sym('x', 1, 1)
#
# interpol_path_x = casadi.interpolant("interpol_spline_x", "bspline", [ref_path_s])
# interp_exp = interpol_path_x(x, p)
# interp_fun = Function('interp_fun', [x, p], [interp_exp])
#
# # test interp_fun
#
# x_test = 0.5
# p_test = np.linspace(0, 1, N_refpath)
# y_test = interp_fun(x_test, p_test)
# print(y_test)
#
# p_test = np.linspace(0, 2, N_refpath)
# y_test = interp_fun(x_test, p_test)
# print(y_test)
#
# # generate C code
# interp_fun.generate()

import casadi as ca


def differentiable_interp(x_points, y_points, values, x_new, y_new):
    """
    Perform differentiable bilinear interpolation manually using CasADi.

    :param x_points: List of numeric x grid points.
    :param y_points: List of numeric y grid points.
    :param values: CasADi symbolic matrix of function values on (x_points, y_points).
    :param x_new: CasADi symbolic list of new x points.
    :param y_new: CasADi symbolic list of new y points.
    :return: Interpolated CasADi expression at (x_new, y_new).
    """

    def find_index(points, new_val):
        return ca.floor(ca.sum1(points <= new_val) - 1)

    x1_idx = find_index(ca.DM(x_points), x_new)
    y1_idx = find_index(ca.DM(y_points), y_new)

    x1 = x_points[x1_idx]
    x2 = x_points[x1_idx + 1]
    y1 = y_points[y1_idx]
    y2 = y_points[y1_idx + 1]

    Q11 = values[x1_idx, y1_idx]
    Q21 = values[x1_idx + 1, y1_idx]
    Q12 = values[x1_idx, y1_idx + 1]
    Q22 = values[x1_idx + 1, y1_idx + 1]

    # Compute bilinear interpolation
    interp_value = ((Q11 * (x2 - x_new) * (y2 - y_new) +
                     Q21 * (x_new - x1) * (y2 - y_new) +
                     Q12 * (x2 - x_new) * (y_new - y1) +
                     Q22 * (x_new - x1) * (y_new - y1)) /
                    ((x2 - x1) * (y2 - y1)))

    return interp_value


# Example usage:
x_points = ca.DM([0, 1, 2])
y_points = ca.DM([0, 1, 2])
values = ca.MX.sym('values', 3, 3)  # Symbolic values at grid points
x_new = ca.MX.sym('x_new')
y_new = ca.MX.sym('y_new')

interp_value = differentiable_interp(x_points, y_points, values, x_new, y_new)


