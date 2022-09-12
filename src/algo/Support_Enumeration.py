from .direct_solver import DirectSolver

import nashpy as nash

# Solve bi-matrix game via NashPy's support-enumeration method.

class SupportEnumeration(DirectSolver):

    def __init__(self, args):
        super(SupportEnumeration, self).__init__(args)

    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "SupportEnumeration now only works for 2-player games!"
        ret = []

        ans = next(nash.Game(utility[0], utility[1]).support_enumeration())
        ret.append(ans[0])
        ret.append(ans[1])

        info = {
            'solver': "SupportEnumeration",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
        }

        return ret, info

    def reset(self):
        pass
