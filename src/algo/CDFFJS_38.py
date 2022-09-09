import numpy as np
import math
from util.lp import solve_lp, solve_zero_sum

sep_const = (3-math.sqrt(5))/2


def solve(R, C):
    # solve zero sum games (R,-R) and (-C,C)
    x_star, y_star, v_r = solve_zero_sum(R, -R)
    x_hat, y_hat, v_c = solve_zero_sum(-C, C)
    if v_c <= v_r:
        if v_r <= sep_const:
            return x_hat, y_star
        else:
            j = np.argmax(x_star @ C, keepdims=True)[0]
            r = np.argmax(R[:, j], keepdims=True)[0]
            e_j = np.zeros_like(y_star)
            e_j[j] = 1.0
            e_r = np.zeros_like(x_star)
            e_r[r] = 1.0
            x_p = 1/(2-v_r) * x_star + (1-v_r)/(2-v_r) * e_r
            return x_p, e_j
    else:
        if v_c <= sep_const:
            return x_hat, y_star
        else:
            i = np.argmax(R @ y_hat, keepdims=True)[0]
            c = np.argmax(C[i, :], keepdims=True)[0]
            e_i = np.zeros_like(x_hat)
            e_i[i] = 1.0
            e_c = np.zeros_like(y_hat)
            e_c[c] = 1.0
            y_p = 1/(2-v_c) * y_hat + (1-v_c)/(2-v_c) * e_c
            return e_i, y_p
