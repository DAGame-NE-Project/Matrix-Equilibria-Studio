import random
class RandomPlay(object):

    def __init__(self, args):
        self.args = args

    def solve(self, game, info):
        actions = game.getActionSpace()
        ret = tuple()
        for player_id in range(game.players):
            ret = ret + tuple([random.randint(0, actions[player_id] - 1)])
        return ret, {"solver": "RandomPlayer"}

    def reset(self):
        pass
