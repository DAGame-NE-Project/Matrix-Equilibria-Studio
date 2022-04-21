import random

class RandomPlay(object):

    def __init__(self, args):
        self.args = args

    def solve(self, game, info):
        actions = game.getActionSpace()
        players = game.players
        ret = [np.zeros(actions[player_id]) for player_id in range(players)]
        for player_id in range(players):
            ret[player_id][random.randint(0, actions[player_id] - 1)] = 1.
        return ret, {"solver": "RandomPlayer"}

    def reset(self):
        pass
