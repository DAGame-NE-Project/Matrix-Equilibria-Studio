import random
class RandomPlay(object):

    def __init__(self):
        pass

    def solve(self, game, info):
        actions = game.getActionSpace()
        ret = tuple()
        for player_id in range(game.players):
            ret = ret + tuple([random.randint(actions[player_id])])
        return ret, {"solver": "RandomPlayer"}

    def reset(self):
        pass
