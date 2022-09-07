import numpy as np
import random

class Player(object):
    def __init__(self, action_len, utility, id):
        """
        :param action_len: 策略个数
        :param utility:  收益矩阵
        :param id: 玩家0写0 玩家1写1
        """
        self.utility = utility
        self.action_len = action_len
        self.action = np.random.random(self.action_len)
        self.id = id


class Nash(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def getActionSpace(self):
        p0action=random.choice(range(self.p0.utility.shape[0]))
        p1action=np.argmax(self.p1.utility[p0action])
        p0action1=np.argmax(self.p0.utility[:,p1action])
        act_p0=np.zeros(self.p0.utility.shape[0])
        act_p1=np.zeros(self.p1.utility.shape[1])
        act_p1[np.ix_([p1action])]=1
        if p0action1!=p0action:
            act_p0[np.ix_([p0action,p0action1])]=0.5
        else:
            act_p0[np.ix_([p0action])]=1
        print('p0', act_p0)
        print('p1', act_p1)



if __name__ == '__main__':

    u0 = np.array(
        [[-4, 0],
         [-5, -1]]
    )
    u1 = np.array(
        [[-4, -5],
         [0, -1]]
    )
    p0 = Player(2, u0, 0)
    p1 = Player(2, u1, 1)
    nash = Nash(p0, p1)
    nash.getActionSpace()
