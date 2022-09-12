import numpy as np
from algo.CDFFJS_38 import *

if __name__ == '__main__':
    from env import matrixgame
    from env.gen.randommatrixgenerator import RandomMatrixGenerator
    import argparse

    def test_RC(R, C):
        args = argparse.Namespace()
        args.players = 2
        args.actionspace = [R.shape[0], R.shape[1]]
        gen = RandomMatrixGenerator(args)
        game = matrixgame.MatrixGame(gen)
        p = Player(args=None)
        print(p.solve(game, [R, C]))

    R = np.array([[0.01, 0, 0], [0.01+0.3393, 1, 1]])
    C = np.array([[0.01, 0.01+0.3393, 0.01+0.3393],
                 [0, 1, 0.812815]])
    test_RC(R, C)
    R, C = C.transpose(), R.transpose()
    test_RC(R, C)
    R = np.array([[0.01, 0, 0], [0.01+0.3393, 1, 1],
                 [0.01+0.3393, 0.582523, 0.582523]])
    C = np.array([[0.01, 0.01+0.3393, 0.01+0.3393],
                 [0, 1, 0.812815], [0, 1, 0.812815]])
    test_RC(R, C)
