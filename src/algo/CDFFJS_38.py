import numpy as np
import math
from util.lp import solve_zero_sum
from .direct_solver import DirectSolver


sep_const = (3-math.sqrt(5))/2


class Player(DirectSolver):
    def __init__(self, args):
        super(Player, self).__init__(args)

    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "CDFFJS_38 only works for 2-player games!"
        ret = []

        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))

        R, C = utility[0], utility[1]
        # solve zero sum games (R,-R) and (-C,C)
        x_star, y_star, v_r = solve_zero_sum(R)
        x_hat, y_hat, v_c = solve_zero_sum(-C)
        if v_c <= v_r:
            if v_r <= sep_const:
                ret[0], ret[1] = x_hat, y_star
            else:
                j = np.argmax(x_star @ C, keepdims=True)[0]
                r = np.argmax(R[:, j], keepdims=True)[0]
                e_j = np.zeros_like(y_star)
                e_j[j] = 1.0
                e_r = np.zeros_like(x_star)
                e_r[r] = 1.0
                x_p = 1/(2-v_r) * x_star + (1-v_r)/(2-v_r) * e_r
                ret[0], ret[1] = x_p, e_j
        else:
            if v_c <= sep_const:
                ret[0], ret[1] = x_hat, y_star
            else:
                i = np.argmax(R @ y_hat, keepdims=True)[0]
                c = np.argmax(C[i, :], keepdims=True)[0]
                e_i = np.zeros_like(x_hat)
                e_i[i] = 1.0
                e_c = np.zeros_like(y_hat)
                e_c[c] = 1.0
                y_p = 1/(2-v_c) * y_hat + (1-v_c)/(2-v_c) * e_c
                ret[0], ret[1] = e_i, y_p
        info = {
            'solver': "CDFFJS_38",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
            'strategy_before_adjust': [x_hat.copy(), y_star.copy()],
        }
        return ret, info

    def reset(self):
        pass


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
