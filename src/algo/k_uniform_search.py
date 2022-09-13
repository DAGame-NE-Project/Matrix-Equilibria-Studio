#from tkinter.filedialog import test

from .direct_solver import DirectSolver
import numpy as np
from util.exhaustive_search import exhaustive_search
import math


class Player(DirectSolver):
    def __init__(self, args):
        super(Player, self).__init__(args)
        self.eps = args.eps if hasattr(args, 'eps') else 0.1
        self.goal = args.goal if hasattr(args, 'goal') else 'epsNE'
        assert self.goal in [
            'epsNE', 'epsWSNE'], "k_uniform_search only works for 'epsNE' and 'epsWSNE'!"

    def solve(self, game, utility):
        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "k_uniform_search only works for 2-player games!"
        ret = []

        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))

        R, C = utility[0], utility[1]
        n = np.max(R.shape)
        k = 1

        if self.goal == 'epsNE':
            k = math.ceil(12 * math.log(n)/self.eps*self.eps)
        else:
            k = math.ceil(2 * math.log(2*n)/self.eps*self.eps)
        ret[0], ret[1] = exhaustive_search(R, C, k, self.goal)
        info = {
            'solver': "k_uniform_search",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
        }
        return ret, info

    def reset(self):
        pass


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
    test_RC(R, C,'epsNE')
    test_RC(R, C,'epsWSNE')
    R = C
    test_RC(R, C, 'epsNE')
    test_RC(R, C, 'epsWSNE')
    # paper-scissors-rock
    R = np.array([[0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5]])
    C = 1 - R
    test_RC(R, C, 'epsNE')
    test_RC(R, C, 'epsWSNE')
