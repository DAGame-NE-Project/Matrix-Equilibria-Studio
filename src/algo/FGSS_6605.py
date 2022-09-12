import numpy as np
from itertools import combinations
from util.lp import solve_lp, solve_zero_sum
from .direct_solver import DirectSolver


class Player(DirectSolver):
    def __init__(self, args):
        super(Player, self).__init__(args)
        if hasattr(args, "NON_ZERO"):
            self.NON_ZERO = args.NON_ZERO
        else:
            self.NON_ZERO = 1e-10


    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "CDFFJS_38 only works for 2-player games!"
        ret = []

        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))

        R, C = utility[0], utility[1]
        strategy_pairs = [None for _ in range(3)]
        epsWSs = [1.0 for _ in range(3)]
        strategy_pairs[0], epsWSs[0] = self.__BestPure(R, C)
        strategy_pairs[1], epsWSs[1] = self.__Best2times2(R, C)
        ret_idx = np.argmin(epsWSs, keepdims=True)[0]
        return strategy_pairs[ret_idx]


    def __BestR(self, S_R, neg_S_C, R):
        # find the best eps-WSNE of y
        action_num_x, action_num_y = R.shape
        # supp(y)\subseteq S_C
        b_eq = np.zeros(len(neg_S_C)+1)
        b_eq[-1] = 1
        A_eq = np.zeros((len(neg_S_C)+1, action_num_y+1))
        A_eq[-1, :-1] = 1
        if len(neg_S_C) > 0:
            A_eq[np.arange(len(neg_S_C)), neg_S_C] = 1
        # constrain eps-WSNE
        A_le = np.zeros((action_num_x, len(S_R), action_num_y))
        A_le -= R[S_R]
        A_le = np.swapaxes(A_le, 0, 1)
        A_le += R
        A_le = np.reshape(A_le, (-1, action_num_y))
        A_le = np.insert(A_le, action_num_y, -1, axis=1)
        b_le = np.zeros(len(A_le))
        c = np.zeros(action_num_y+1)
        c[-1] = 1
        lp_sol = solve_lp(c=c, A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq)
        if lp_sol.success:
            return lp_sol.x[:-1], lp_sol.x[-1]


    def __BestC(self, S_C, neg_S_R, C):
        # find the best eps-WSNE of x
        action_num_x, action_num_y = C.shape
        # supp(x)\subseteq S_R
        b_eq = np.zeros(len(neg_S_R)+1)
        b_eq[-1] = 1
        A_eq = np.zeros((len(neg_S_R)+1, action_num_x+1))
        A_eq[-1, :-1] = 1
        if len(neg_S_R) > 0:
            A_eq[np.arange(len(neg_S_R)), neg_S_R] = 1
        # constrain eps-WSNE
        A_le = np.zeros((action_num_y, len(S_C), action_num_x))
        CT = np.transpose(C)
        A_le -= CT[S_C]
        A_le = np.swapaxes(A_le, 0, 1)
        A_le += CT
        A_le = np.reshape(A_le, (-1, action_num_x))
        A_le = np.insert(A_le, action_num_x, -1, axis=1)
        b_le = np.zeros(len(A_le))
        c = np.zeros(action_num_x+1)
        c[-1] = 1
        lp_sol = solve_lp(c=c, A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq)
        if lp_sol.success:
            return lp_sol.x[:-1], lp_sol.x[-1]


    def __BestPure(self, R, C):
        # find the best eps-WSNE of pure strategy pairs
        def regret(i, j): return max(
            np.max(R[:, j])-R[i, j], np.max(C[i, :]-C[i, j]))
        action_num_x, action_num_y = R.shape
        ret_x, ret_y = None, None
        epsWS = 1.0
        for i in range(action_num_x):
            for j in range(action_num_y):
                new_regret = regret(i, j)
                if new_regret <= epsWS:
                    ret_x = np.zeros(action_num_x)
                    ret_x[i] = 1.0
                    ret_y = np.zeros(action_num_y)
                    ret_y[j] = 1.0
                    epsWS = new_regret
        return (ret_x, ret_y), epsWS


    def __Best2times2(self, R, C):
        # find the best eps-WSNE of 2x2 subgames
        action_num_x, action_num_y = R.shape
        ret_x, ret_y = None, None
        epsWS = 1.0
        for xi in combinations(range(action_num_x), 2):
            for yi in combinations(range(action_num_y), 2):
                S_R = np.array(xi)
                neg_S_R = np.setdiff1d(
                    np.arange(action_num_x), S_R, assume_unique=True)
                S_C = np.array(yi)
                neg_S_C = np.setdiff1d(
                    np.arange(action_num_y), S_C, assume_unique=True)
                new_x, new_regret_x = self.__BestC(S_C, neg_S_R, C)
                new_y, new_regret_y = self.__BestR(S_R, neg_S_C, R)
                new_regret = max(new_regret_x, new_regret_y)
                if new_regret <= epsWS:
                    ret_x = new_x
                    ret_y = new_y
                    epsWS = new_regret
        return (ret_x, ret_y), epsWS


    def __BestZeroSum(self, R, C):
        # find the best eps-WSNE of shifting probability of NE of (D,-D)
        action_num_x, action_num_y = R.shape
        x, y, _ = solve_zero_sum((R-C)/2)
        S_R = np.argwhere(x > self.NON_ZERO)
        neg_S_R = np.setdiff1d(
            np.arange(action_num_x), S_R, assume_unique=True)
        S_C = np.argwhere(y > self.NON_ZERO)
        neg_S_C = np.setdiff1d(
            np.arange(action_num_y), S_C, assume_unique=True)
        ret_x, regret_x = self.__BestC(S_C, neg_S_R, C)
        ret_y, regret_y = self.__BestR(S_R, neg_S_C, R)
        epsWS = max(regret_x, regret_y)
        return (ret_x, ret_y), epsWS


    def test(self):
        # test BestR
        # extended matching pennies
        R = np.array([[0, 1, 0], [1, 0, 1]])
        S_R = np.array([0, 1])
        neg_S_C = np.array([])
        print(self.__BestR(S_R, neg_S_C, R))

        C = 1 - R
        S_C = np.array([0, 1, 2])
        neg_S_R = np.array([])
        print(self.__BestC(S_C, neg_S_R, C))

        # test paper-scissors-rock
        R = np.array([[0.5, 1, 0], [0, 0.5, 1], [1, 0, 0.5]])
        C = 1 - R
        S_R = np.array([0, 1])
        S_C = np.array([0, 1])
        neg_S_R = np.array([2])
        neg_S_C = np.array([2])
        print(self.__BestR(S_R, neg_S_C, R))
        print(self.__BestC(S_C, neg_S_R, C))

        S_R = np.array([0, 1, 2])
        S_C = np.array([0, 1, 2])
        neg_S_R = np.array([])
        neg_S_C = np.array([])
        print(self.__BestR(S_R, neg_S_C, R))
        print(self.__BestC(S_C, neg_S_R, C))

        # test BestPure
        # prisoner's dilemma
        R = np.array([[1/2, 0], [1, 2/3]])
        C = np.transpose(R)
        print(self.__BestPure(R, C))

        # test Best2times2
        # extended matching pennies
        R = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]])
        print(self.__Best2times2(R, C))

        # test shifting probability
        R = np.array([[1/3-0.01, 1], [0, 0]])
        C = np.array([[1, 1/3-0.01], [0, 0]])
        print((R-C)/2, self.__BestZeroSum(R, C))


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
