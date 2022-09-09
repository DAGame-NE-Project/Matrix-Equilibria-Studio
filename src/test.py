import numpy as np
from algo.KS_2_3 import *
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
    R = np.array([[1, 1/3-0.01], [1/3-0.01, 1],[0,0]])
    C = np.array([[1/3-0.01, 1], [1, 1/3-0.01],[0,0]])
    print((R-C)/2, solve(R, C))