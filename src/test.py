import numpy as np
from algo.TS import *

if __name__ == '__main__':
    # test tight instance of DFM
    from env import matrixgame
    from env.gen.randommatrixgenerator import RandomMatrixGenerator
    import argparse
    args = argparse.Namespace()
    eps = 1e-6
    R = np.array([[0, 0, 0], [1/3, 1, 1], [1/3, 2/3-eps/2, 2/3-eps/2]])
    C = np.array([[0, 1/3-eps, 1/3-eps], [0, 1, 2/3+eps], [0, 1, 2/3+eps]])
    args.players = 2
    args.actionspace = [R.shape[0], R.shape[1]]
    gen = RandomMatrixGenerator(args)
    game = matrixgame.MatrixGame(gen)
    init_x = np.array([1.0, 0, 0])
    init_y = np.array([1.0, 0, 0])
    args.init_x = init_x
    args.init_y = init_y
    p = Player(args=args)
    x, y = p.solve(game, [R, C])
    print(x, y)
    print(calculate_f_value(R, C, x[0], x[1]))
    R, C = C.transpose(), R.transpose()
    x, y = p.solve(game, [R, C])
    print(x, y)
    print(calculate_f_value(R, C, x[0], x[1]))
