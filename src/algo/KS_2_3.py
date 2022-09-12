from util.lp import solve_zero_sum
import numpy as np
from .direct_solver import DirectSolver


class Player(DirectSolver):
    def __init__(self, args):
        super(Player, self).__init__(args)

    def solve(self, game, utility):
        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "KS_2_3 only works for 2-player games!"
        ret = []
        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))
        R, C = utility[0], utility[1]
        # check if there is a pure strategy with 2/3-WSNE
        action_num_x, action_num_y = R.shape
        x_sol = np.zeros(action_num_x)
        y_sol = np.zeros(action_num_y)
        for i in range(action_num_x):
            for j in range(action_num_y):
                if R[i, j] >= 1/3 and C[i, j] >= 1/3:
                    x_sol[i] = 1
                    y_sol[j] = 1
                    break
            else:
                continue
            break
        else:
            # otherwise, solve a zero sum game:
            x_sol, y_sol, _ = solve_zero_sum((R-C)/2)
        ret[0], ret[1] = x_sol, y_sol
        info = {
            'solver': "KS_2_3",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
        }
        return ret, info

    def reset(self):
        pass


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
