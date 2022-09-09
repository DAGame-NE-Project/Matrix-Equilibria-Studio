import numpy as np
import numpy as np
from algo.CDFFJS_6528 import *

if __name__ == '__main__':
    R = np.array([[0.01, 0, 0], [0.01+0.3393, 1, 1]])
    C = np.array([[0.01, 0.01+0.3393, 0.01+0.3393],
                 [0, 1, 0.812815]])
    print(solve(R, C))
    R, C = C.transpose(), R.transpose()
    print(solve(R, C))
    R = np.array([[0.01, 0, 0], [0.01+0.3393, 1, 1],
                 [0.01+0.3393, 0.582523, 0.582523]])
    C = np.array([[0.01, 0.01+0.3393, 0.01+0.3393],
                 [0, 1, 0.812815], [0, 1, 0.812815]])
    print(solve(R, C))

    # test other cases
    R = np.array([[0.4, 0, 0], [0.4+0.3393, 1, 1]])
    C = np.array([[0.4, 0.4+0.3393, 0.4+0.3393],
                 [0, 1, 0.812815]])
    print(solve(R, C))
    R, C = C.transpose(), R.transpose()
    print(solve(R, C))
    R = np.array([[0.4, 0, 0], [0.4+0.3393, 1, 1],
                 [0.4+0.3393, 0.582523, 0.582523]])
    C = np.array([[0.4, 0.4+0.3393, 0.4+0.3393],
                 [0, 1, 0.812815], [0, 1, 0.812815]])
    print(solve(R, C))
    R, C = C.transpose(), R.transpose()
    print(solve(R, C))