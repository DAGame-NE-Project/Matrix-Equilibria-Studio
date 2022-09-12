import numpy as np
from algo.KS_2_3 import *

if __name__ == '__main__':
    # test solver
    from env import matrixgame
    from env.gen.randommatrixgenerator import RandomMatrixGenerator
    import argparse

    def test_RC(R, C):
        args = argparse.Namespace()
        args.players = 2
        args.actionspace = [R.shape[0], R.shape[1]]
        gen = RandomMatrixGenerator(args)
        game = matrixgame.MatrixGame(gen)
        p = Player(args=argparse.Namespace())
        print(p.solve(game, [R, C]))

    # pure 2/3-WSNE
    R = np.array([[1, 1/3], [1, 0], [0, 0]])
    C = np.array([[0, 1/3], [0, 1], [0, 0]])
    test_RC(R, C)
    # not pure (2/3)-WSNE
    R = np.array([[1, 1/4], [1, 0], [0, 0]])
    C = np.array([[0, 1/4], [0, 1], [0, 0]])
    print((R-C)/2)
    test_RC(R, C)
    # tight 2/3-WSNE
    R = np.array([[1, 1/3-0.01], [0, 0]])
    C = np.array([[1/3-0.01, 1], [0, 0]])
    print((R-C)/2)
    test_RC(R, C)

    # another tight 2/3-WSNE
    R = np.array([[1, 1/3-0.01], [1/3-0.01, 1], [0, 0]])
    C = np.array([[1/3-0.01, 1], [1, 1/3-0.01], [0, 0]])
    print((R-C)/2)
    test_RC(R, C)
