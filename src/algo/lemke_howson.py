from .direct_solver import DirectSolver

import nashpy as nash

# Solve bi-matrix game via NashPy's Lemke-Howson method.

class LemkeHowson(DirectSolver):

    def __init__(self, args):
        super(LemkeHowson, self).__init__(args)

    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "LemkeHowson now only works for 2-player games!"
        ret = []

        ans = nash.Game(utility[0], utility[1]).lemke_howson(initial_dropped_label=0)
        ret.append(ans[0])
        ret.append(ans[1])

        info = {
            'solver': "Lemke-Howson",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
        }

        return ret, info

    def reset(self):
        pass

