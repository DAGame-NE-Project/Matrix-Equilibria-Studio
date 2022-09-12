import abc

class DirectSolver(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        self.args = args

    @abc.abstractmethod
    def solve(self, game, utility):
        # returns (mixed strategy, info)
        pass

    @abc.abstractmethod
    def reset(self):
        pass

