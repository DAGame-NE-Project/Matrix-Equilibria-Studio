import numpy as np
from algo.FGSS_6605 import *

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
        p = Player(args=argparse.Namespace())
        p.test()
        print(p.solve(game, [R, C]))

        # test all
    R = np.array([[0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5]])
    C = 1 - R
    test_RC(R, C)
