from copy import deepcopy

class MatrixGame(object):

    def __init__(self, generator):
        self.generator = generator
        self.reset()

    def getActionSpace(self):
        return deepcopy(self.actionspace)

    def getActionSpaceID(self, ID):
        return deepcopy(self.actionspace[ID])

    def play(a):
        return deepcopy(U(a))

    def playID(a, ID):
        return deepcopy(U(a)[ID])

    def reset(self):
        self.players, self.actionspace, self.U = self.generator()

