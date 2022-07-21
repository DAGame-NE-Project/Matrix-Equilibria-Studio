import numpy as np
from IPython import embed


class ReplicatorDynamics(object):
    def __init__(self, args):
        self.args = args
        self.game_u = None

    def get_game_utility(self, game):
        n_player = game.players
        action_space_n = game.actionspace
        game_u = np.zeros(shape=[*action_space_n, n_player])  # Before transpose
        a = [0] * n_player

        def dfs(i):
            if i == n_player:
                game_u[tuple(a)] = game.U(a)
                return
            for action in range(action_space_n[i]):
                a[i] = action
                dfs(i + 1)
                a[i] = -1
        dfs(0)
        game_u = np.transpose(game_u, [n_player, *list(range(n_player))])
        return game_u

    def solve(self, game, info):
        n_player = game.players
        action_space_n = game.actionspace

        if (info is None or info['solver'] != "ReplicatorDynamics"
                or self.game_u is None):
            init_strategy_n = [np.ones(action_space_n[i]) / action_space_n[i]
                               for i in range(n_player)]
            init_overall_policy = [np.zeros(action_space_n[i]) for i in range(n_player)]
            self.game_u = self.get_game_utility(game)
            info = {
                'solver': "ReplicatorDynamics",
                'strategy_n': init_strategy_n,
                'overall_policy': init_overall_policy,
            }
        game_u = self.game_u
        strategy_n = info['strategy_n']

        def reshape_strategy(s, n, i):
            shape = [1 for _ in range(n)]
            shape[i] = len(s)
            s = s.reshape(shape)
            return s

        delta_n = []
        for i in range(n_player):
            util_vec = game_u[i]
            for j in range(n_player):
                if j != i:
                    s_j = reshape_strategy(strategy_n[j], n_player, j)
                    util_vec = np.sum(util_vec * s_j, axis=j, keepdims=True)
            util_vec = util_vec.reshape(action_space_n[i])
            util = np.sum(util_vec * strategy_n[i])
            delta_i = (util_vec - util) * strategy_n[i]
            delta_n.append(delta_i)
        for i in range(n_player):
            strategy_n[i] += delta_n[i]

        info['strategy_n'] = strategy_n
        for i in range(n_player):
            info['overall_policy'][i] += strategy_n[i]

        current_strategy, current_action, current_info = strategy_n, None, info
        return current_strategy, current_action, current_info

    def reset(self):
        pass
