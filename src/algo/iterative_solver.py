import abc

class IterativeSolver(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def step(self, game, info):
        pass

    @abstractmethod
    def reset(self):
        pass

