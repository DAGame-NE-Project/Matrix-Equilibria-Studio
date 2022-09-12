import numpy as np
import math
from util.lp import solve_lp, solve_zero_sum
from util.exhaustive_search import exhaustive_search
from .direct_solver import DirectSolver


class Player(DirectSolver):
    def __init__(self, args):
        super(Player, self).__init__(args)
        if hasattr(args, "delta"):
            self.delta = args.delta
        else:
            self.delta = 0.1
        if hasattr(args, "NON_ZERO"):
            self.NON_ZERO = args.NON_ZERO
        else:
            self.NON_ZERO = 1e-10

    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "DFM_50 only works for 2-player games!"
        ret = []

        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))

        R, C = utility[0], utility[1]

        kappa = math.ceil(2*math.log(1/self.delta)/(self.delta*self.delta))
        # solve zero-sum games
        action_num_x, action_num_y = R.shape
        x_star, y_star, _ = solve_zero_sum(R)
        x_hat, y_hat, _ = solve_zero_sum(-C)
        if x_star @ R @ y_star >= x_hat @ C @ y_hat:
            if x_star @ R @ y_star <= 1/2:
                ret[0], ret[1] = x_hat, y_star
            else:
                # search for a strategy x_p by a linear program
                # supp(x_p)\subseteq supp(x_star)
                nonsupp_x_star = np.argwhere(x_star <= NON_ZERO)
                b_eq = np.zeros(len(nonsupp_x_star)+1)
                b_eq[-1] = 1
                A_eq = np.zeros((len(nonsupp_x_star)+1, action_num_x))
                A_eq[-1] = 1
                if len(nonsupp_x_star) > 0:
                    A_eq[np.arange(len(nonsupp_x_star)), nonsupp_x_star] = 1
                A_le = np.transpose(C)
                b_le = np.ones(action_num_y)*0.5
                c = np.zeros(action_num_x)
                lp_sol = solve_lp(c=c, A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq)
                if lp_sol.success:
                    ret[0], ret[1] = lp_sol.x, y_star
                # do an exhaustive search
                else:
                    ret[0], ret[1] = exhaustive_search(R, C, kappa, 'epsWSNE')
        else:
            # symmetric case
            if x_hat @ C @ y_hat <= 1/2:
                ret[0], ret[1] = x_hat, y_star
            # search for a strategy y_p by a linear program
            # supp(y_p)\subseteq supp(y_star)
            else:
                nonsupp_y_hat = np.argwhere(y_hat <= NON_ZERO)
                b_eq = np.zeros(len(nonsupp_y_hat)+1)
                b_eq[-1] = 1
                A_eq = np.zeros((len(nonsupp_y_hat)+1, action_num_y))
                A_eq[-1] = 1
                if len(nonsupp_y_hat) > 0:
                    A_eq[np.arange(len(nonsupp_y_hat)), nonsupp_y_hat] = 1
                A_le = R
                b_le = np.ones(action_num_x)*0.5
                c = np.zeros(action_num_y)

                if lp_sol.success:
                    ret[0], ret[1] = x_hat, lp_sol.x
                # do an exhaustive search
                else:
                    ret[0], ret[1] = exhaustive_search(R, C, kappa, 'epsWSNE')
        info ={
            "solver": "DFM_50",
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
    # extended matching pennies
    R = np.array([[0, 1, 0], [1, 0, 1]])
    C = 1 - R
    test_RC(R, C)

    # paper-scissors-rock
    R = np.array([[0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5]])
    C = 1 - R
    test_RC(R, C)
