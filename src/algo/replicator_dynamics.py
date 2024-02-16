from .iterative_solver import IterativeSolver
from util import sample_utility

import nashpy as nash
import numpy as np

# Solve bi-matrix game via NashPy's asymmetric-replicator-dynamics method.

class ReplicatorDynamics(IterativeSolver):

    def __init__(self, args):
        super(ReplicatorDynamics, self).__init__(args)
        self.reset()

    def step(self, game, info):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "ReplicatorDynamics now only works for 2-player games!"
        if self.tmp_game is None or self.tmp_game is not game:
            self.tmp_game = game
            utility = sample_utility(game, 1)
            utility = utility.transpose((len(utility.shape) - 1, ) + tuple(range(0, len(utility.shape) - 1)))
            self.nashpy_game = nash.Game(utility[0], utility[1])
        if info is None or info['solver'] != "ReplicatorDynamics":
            info = {
                'solver': "ReplicatorDynamics",
                'agents_population': [[1.0 / actions[player_id] for action_id in range(actions[player_id])] for player_id in range(players)],
                'overall_policy': [[0.0 for action_id in range(actions[player_id])] for player_id in range(players)],
            }

        ret = []
        tmp_ret = tuple()
        for player_id in range(players):
            policy = np.array(info['agents_population'][player_id])
            policy = np.maximum(policy, 0)
            policy = policy / np.sum(policy)
            # set negative part zero
            ret.append(policy)
            tmp_ret = tmp_ret + tuple([int(np.random.choice(np.array(range(actions[player_id])), p = policy))])

        self._update_info(game, tmp_ret, ret, info)

        for player_id in range(players):
            info['overall_policy'][player_id] += ret[player_id]

        return ret, tmp_ret, info

    def reset(self):
        self.tmp_game = None
        self.nashpy_game = None

    def _update_info(self, game, ret, policy, info):
        xs, ys = self.nashpy_game.asymmetric_replicator_dynamics(x0=info['agents_population'][0], y0=info['agents_population'][1], timepoints=np.array([0,1]))
        info['agents_population'] = [xs[1], ys[1]]
