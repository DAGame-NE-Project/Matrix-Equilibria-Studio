import numpy as np
import math
from util.lp import solve_lp, solve_zero_sum

z = 0.013906376


def cal_epsWS(R, C, xi, yi):
    return max(np.max(R[:, yi])-R[xi, yi], np.max(C[xi, :])-C[xi, yi])


def solve(R, C, NON_ZERO=1e-10):
    # solve zero sum games (R,-R) and (-C,C)
    x_star, y_star, v_r = solve_zero_sum(R, -R)
    x_hat, y_hat, v_c = solve_zero_sum(-C, C)
    if v_c <= v_r:
        if v_r <= 2/3-z:
            return x_hat, y_star
        elif np.max(x_star @ C) <= 2/3-z:
            return x_star, y_star
        else:
            j_star = np.argmax(x_star @ C, keepdims=True)[0]
            e_j_star = np.zeros_like(y_star)
            e_j_star[j_star] = 1.0
            supp_x_star = np.argwhere(x_star > NON_ZERO)
            S = np.intersect1d(supp_x_star, np.argwhere(R[:, j_star] < 1/3+z))
            B = np.setdiff1d(np.arange(len(x_star)), S)
            x_B = x_star.copy()
            x_B[S] = 0.0
            x_B = x_B/np.sum(x_B)
            if x_B @ C[:, j_star] >= 1/3+z:
                return x_B, e_j_star
            else:
                j_p = np.argmax(x_B @ C, keepdims=True)[0]
                e_j_p = np.zeros_like(y_star)
                e_j_p[j_p] = 1.0
                for i in np.nditer(supp_x_star):
                    if cal_epsWS(R, C, i, j_star) <= 2/3-z:
                        e_i = np.zeros_like(x_star)
                        e_i[i] = 1.0
                        return e_i, e_j_star
                    elif cal_epsWS(R, C, i, j_p) <= 2/3-z:
                        e_i = np.zeros_like(x_star)
                        e_i[i] = 1.0
                        return e_i, e_j_p

                b = B.copy()
                b = np.intersect1d(b, np.argwhere((R[:, j_star] > 1 - 18*z/(1+3*z))
                                                  & (C[:, j_p] > 1-18*z/(1+3*z))))
                s = S.copy()
                s = np.intersect1d(s, np.argwhere(
                    (R[:, j_star] > 1 - 27*z/(1+3*z)) & (C[:, j_p] > 1-27*z/(1+3*z))))

                x_mp = np.zeros_like(x_star)
                x_mp[b] = (1-24*z)/(2-39*z)
                x_mp[s] = (1-15*z)/(2-39*z)

                y_mp = np.zeros_like(y_star)
                y_mp[j_star] = (1-24*z)/(2-39*z)
                y_mp[j_p] = (1-15*z)/(2-39*z)
                return x_mp, y_mp
    else:
        if v_c <= 2/3-z:
            return x_hat, y_star
        elif np.max(R @ y_hat) <= 2/3-z:
            return x_hat, y_hat
        else:
            i_hat = np.argmax(R @ y_hat, keepdims=True)[0]
            e_i_hat = np.zeros_like(x_hat)
            e_i_hat[i_hat] = 1.0
            supp_y_hat = np.argwhere(y_hat > NON_ZERO)
            S = np.intersect1d(supp_y_hat, np.argwhere(C[i_hat, :] < 1/3+z))
            B = np.setdiff1d(np.arange(len(y_hat)), S)
            y_B = y_hat.copy()
            y_B[S] = 0.0
            y_B = y_B/np.sum(y_B)
            if R[i_hat, :] @ y_B >= 1/3+z:
                return e_i_hat, y_B
            else:
                i_p = np.argmax(R @ y_B, keepdims=True)[0]
                e_i_p = np.zeros_like(x_hat)
                e_i_p[i_p] = 1.0
                for j in np.nditer(supp_y_hat):
                    if cal_epsWS(R, C, i_hat, j) <= 2/3-z:
                        e_j = np.zeros_like(y_hat)
                        e_j[j] = 1.0
                        return e_i_hat, e_j
                    elif cal_epsWS(R, C, i_p, j) <= 2/3-z:
                        e_j = np.zeros_like(y_hat)
                        e_j[j] = 1.0
                        return e_i_p, e_j

                b = B.copy()
                b = np.intersect1d(b, np.argwhere((R[i_hat, :] > 1 - 18*z/(1+3*z))
                                                  & (C[i_p, :] > 1-18*z/(1+3*z))))
                s = S.copy()
                s = np.intersect1d(s, np.argwhere(
                    (R[i_hat, :] > 1 - 27*z/(1+3*z)) & (C[i_p, :] > 1-27*z/(1+3*z))))

                x_mp = np.zeros_like(x_hat)
                x_mp[i_hat] = (1-24*z)/(2-39*z)
                x_mp[i_p] = (1-15*z)/(2-39*z)

                y_mp = np.zeros_like(y_hat)
                y_mp[b] = (1-24*z)/(2-39*z)
                y_mp[s] = (1-15*z)/(2-39*z)
                return x_mp, y_mp
