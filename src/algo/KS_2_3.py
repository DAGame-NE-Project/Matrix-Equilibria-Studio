from util.lp import solve_zero_sum
import numpy as np


def solve(R, C):
    # check if there is a pure strategy with 2/3-WSNE
    action_num_x, action_num_y = R.shape
    x_sol = np.zeros(action_num_x)
    y_sol = np.zeros(action_num_y)
    for i in range(action_num_x):
        for j in range(action_num_y):
            if R[i, j] >= 1/3 and C[i, j] >= 1/3:
                x_sol[i] = 1
                y_sol[j] = 1
                return x_sol, y_sol
    # otherwise, solve a zero sum game:
    x_sol, y_sol, _ = solve_zero_sum((R-C)/2)
    return x_sol, y_sol


if __name__ == '__main__':
    # test solver
    # pure 2/3-WSNE
    R = np.array([[1, 1/3], [1, 0], [0, 0]])
    C = np.array([[0, 1/3], [0, 1], [0, 0]])
    print(solve(R, C))
    # not pure (2/3)-WSNE
    R = np.array([[1, 1/4], [1, 0], [0, 0]])
    C = np.array([[0, 1/4], [0, 1], [0, 0]])
    print((R-C)/2, solve(R, C))
    # tight 2/3-WSNE
    R = np.array([[1, 1/3-0.01], [0, 0]])
    C = np.array([[1/3-0.01, 1], [0, 0]])
    print((R-C)/2, solve(R, C))

    # another tight 2/3-WSNE
    R = np.array([[1, 1/3-0.01], [1/3-0.01, 1], [0, 0]])
    C = np.array([[1/3-0.01, 1], [1, 1/3-0.01], [0, 0]])
    print((R-C)/2, solve(R, C))
