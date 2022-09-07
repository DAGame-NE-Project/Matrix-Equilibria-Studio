import numpy as np
from itertools import combinations
from util.lp import solve_lp, solve_zero_sum


def solve(R, C, NON_ZERO=1e-10):
    strategy_pairs = [None for _ in range(3)]
    epsWSs = [1.0 for _ in range(3)]
    strategy_pairs[0], epsWSs[0] = BestPure(R, C)
    strategy_pairs[1], epsWSs[1] = Best2times2(R, C)
    strategy_pairs[2], epsWSs[2] = BestZeroSum(R, C, NON_ZERO)
    ret_idx = np.argmin(epsWSs)[0]
    return strategy_pairs[ret_idx]


def BestR(S_R, neg_S_C, R):
    # find the best eps-WSNE of y
    action_num_x, action_num_y = R.shape
    # supp(y)\subseteq S_C
    b_eq = np.zeros(neg_S_C+1)
    b_eq[-1] = 1
    A_eq = np.zeros((neg_S_C+1, action_num_y+1))
    A_eq[-1, :-1] = 1
    A_eq[np.arange(neg_S_C), neg_S_C] = 1
    # constrain eps-WSNE
    A_le = np.zeros((action_num_x, len(S_R), action_num_y))
    A_le -= R[S_R]
    np.swapaxes(A_le, 0, 1)
    A_le += R
    A_le = np.reshape(A_le, (-1, action_num_y))
    A_le = np.insert(A_le, action_num_y, -1)
    b_le = np.zeros(len(A_le))
    c = np.zeros(action_num_y+1)
    c[-1] = 1
    lp_sol = solve_lp(c=c, A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq)
    if lp_sol.success:
        return lp_sol.x[:-1], lp_sol.x[-1]


def BestC(S_C, neg_S_R, C):
    # find the best eps-WSNE of x
    action_num_x, action_num_y = C.shape
    # supp(x)\subseteq S_R
    b_eq = np.zeros(neg_S_R+1)
    b_eq[-1] = 1
    A_eq = np.zeros((neg_S_R+1, action_num_x+1))
    A_eq[-1, :-1] = 1
    A_eq[np.arange(neg_S_R), neg_S_R] = 1
    # constrain eps-WSNE
    A_le = np.zeros((action_num_y, len(S_C), action_num_x))
    CT = np.transpose(C)
    A_le -= CT[S_C]
    np.swapaxes(A_le, 0, 1)
    A_le += CT
    A_le = np.reshape(A_le, (-1, action_num_x))
    A_le = np.insert(A_le, action_num_x, -1)
    b_le = np.zeros(len(A_le))
    c = np.zeros(action_num_x+1)
    c[-1] = 1
    lp_sol = solve_lp(c=c, A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq)
    if lp_sol.success:
        return lp_sol.x[:-1], lp_sol.x[-1]


def BestPure(R, C):
    # find the best eps-WSNE of pure strategy pairs
    def regret(i, j): return max(
        np.max(R[:, j])-R[i, j], np.max(C[i, :]-C[i, j]))
    action_num_x, action_num_y = R.shape
    ret_x, ret_y = None, None
    epsWS = 1.0
    for i in range(len(action_num_x)):
        for j in range(len(action_num_y)):
            new_regret = regret(i, j)
            if new_regret <= epsWS:
                ret_x = np.zeros(action_num_x)
                ret_x[i] = 1.0
                ret_y = np.zeros(action_num_y)
                ret_y[j] = 1.0
                epsWS = new_regret
    return (ret_x, ret_y), epsWS


def Best2times2(R, C):
    # find the best eps-WSNE of 2x2 subgames
    action_num_x, action_num_y = R.shape
    ret_x, ret_y = None, None
    epsWS = 1.0
    for xi in combinations(range(action_num_x), 2):
        for yi in combinations(range(action_num_y), 2):
            S_R = np.array(xi)
            neg_S_R = np.setdiff1d(
                np.arange(action_num_x), S_R, assume_unique=True)
            S_C = np.array(yi)
            neg_S_C = np.setdiff1d(
                np.arange(action_num_y), S_C, assume_unique=True)
            new_x, new_regret_x = BestC(S_C, neg_S_R, C)
            new_y, new_regret_y = BestR(S_R, neg_S_C, R)
            new_regret = max(new_regret_x, new_regret_y)
            if new_regret <= epsWS:
                ret_x = new_x
                ret_y = new_y
                epsWS = new_regret
    return (ret_x, ret_y), epsWS


def BestZeroSum(R, C, NON_ZERO=1e-10):
    # find the best eps-WSNE of shifting probability of NE of (D,-D)
    action_num_x, action_num_y = R.shape
    x, y, _ = solve_zero_sum((R-C)/2)
    S_R = np.argwhere(x > NON_ZERO)
    neg_S_R = np.setdiff1d(
        np.arange(action_num_x), S_R, assume_unique=True)
    S_C = np.argwhere(y > NON_ZERO)
    neg_S_C = np.setdiff1d(
        np.arange(action_num_y), S_C, assume_unique=True)
    ret_x, regret_x = BestC(S_C, neg_S_R, C)
    ret_y, regret_y = BestR(S_R, neg_S_C, R)
    epsWS = max(regret_x, regret_y)
    return (ret_x, ret_y), epsWS
