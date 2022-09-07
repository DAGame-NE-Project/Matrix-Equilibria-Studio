import numpy as np
import math
from util.lp import solve_lp, solve_zero_sum
from util.exhaustive_search import exhaustive_search


def solve(R, C, delta=0.1, NON_ZERO=1e-10):
    kappa = math.ceil(2*math.log(1/delta)/(delta*delta))
    # solve zero-sum games
    action_num_x, action_num_y = R.shape
    x_star, y_star, _ = solve_zero_sum(R)
    x_hat, y_hat, _ = solve_zero_sum(-C)
    if x_star @ R @ y_star >= x_hat @ C @ y_hat:
        if x_star @ R @ y_star <= 1/2:
            return x_hat, y_star
        # search for a strategy x_p by a linear program
        # supp(x_p)\subseteq supp(x_star)
        nonsupp_x_star = np.argwhere(x_star <= NON_ZERO)
        b_eq = np.zeros(len(nonsupp_x_star)+1)
        b_eq[-1] = 1
        A_eq = np.zeros((len(nonsupp_x_star)+1, action_num_x))
        A_eq[-1] = 1
        A_eq[np.arange(len(nonsupp_x_star)), nonsupp_x_star] = 1
        A_le = np.transpose(C)
        b_le = np.ones(action_num_y)*0.5
        c = np.zeros(action_num_x)
        lp_sol = solve_lp(c=c, A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq)
        if lp_sol.success:
            return lp_sol.x, y_star
        # do an exhaustive search
        return exhaustive_search(R, C, kappa, 'epsWSNE')
    else:
        # symmetric case
        if x_hat @ C @ y_hat <= 1/2:
            return x_hat, y_star
        # search for a strategy y_p by a linear program
        # supp(y_p)\subseteq supp(y_star)
        nonsupp_y_hat = np.argwhere(y_hat <= NON_ZERO)
        b_eq = np.zeros(len(nonsupp_y_hat)+1)
        b_eq[-1] = 1
        A_eq = np.zeros((len(nonsupp_y_hat)+1, action_num_y))
        A_eq[-1] = 1
        A_eq[np.arange(len(nonsupp_y_hat)), nonsupp_y_hat] = 1
        A_le = R
        b_le = np.ones(action_num_x)*0.5
        c = np.zeros(action_num_y)

        if lp_sol.success:
            return x_hat, lp_sol.x
        # do an exhaustive search
        return exhaustive_search(R, C, kappa, 'epsWSNE')
