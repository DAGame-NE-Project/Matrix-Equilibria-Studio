import numpy as np
import numpy as np
from algo.TS import *

if __name__ == '__main__':
    # test tight instance
    eps = 1e-6
    R = np.array([[0, 0, 0], [1/3, 1, 1], [1/3, 2/3-eps/2, 2/3-eps/2]])
    C = np.array([[0, 1/3-eps, 1/3-eps], [0, 1, 2/3+eps], [0, 1, 2/3+eps]])
    init_x = np.array([1.0, 0, 0])
    init_y = np.array([1.0, 0, 0])
    x, y = solve(R, C, adjust_method='DFM',
                 line_search_method='adapted', init_x=init_x, init_y=init_y)
    print(calculate_f_value(R, C, x, y))
    R, C = C.transpose(), R.transpose()
    x, y = solve(R, C, adjust_method='DFM',
                 line_search_method='adapted', init_x=init_x, init_y=init_y)
    print(calculate_f_value(R, C, x, y))