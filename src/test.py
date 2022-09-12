import numpy as np
from algo.k_uniform_search import *

if __name__ == '__main__':
    from env import matrixgame
    from env.gen.randommatrixgenerator import RandomMatrixGenerator
    import argparse

    def test_RC(R, C, goal):
        args = argparse.Namespace()
        args.players = 2
        args.actionspace = [R.shape[0], R.shape[1]]
        gen = RandomMatrixGenerator(args)
        game = matrixgame.MatrixGame(gen)
        args.goal = goal
        p = Player(args=args)
        print(p.solve(game, [R, C]))

    R = np.array([[1, 0, 0], [0, 1, 0]])
    C = np.array([[0, 1, 0], [1, 0, 0]])
    # test exhaustive_search
    test_RC(R, C, 'epsNE')
    test_RC(R, C, 'epsWSNE')
    R = C
    test_RC(R, C, 'epsNE')
    test_RC(R, C, 'epsWSNE')
    # paper-scissors-rock
    R = np.array([[0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5]])
    C = 1 - R
    test_RC(R, C, 'epsNE')
    test_RC(R, C, 'epsWSNE')
