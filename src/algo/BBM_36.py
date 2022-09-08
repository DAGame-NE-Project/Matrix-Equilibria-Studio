import numpy as np


class Player(object):
    def __init__(self, action_len, utility, id, action):
        """
        :param action_len: 策略个数
        :param utility:  收益矩阵
        :param id: 玩家0写0 玩家1写1
        """
        self.utility = utility
        self.action_len = action_len
        self.action=action
        self.id = id

  
class Nash(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def getIncentive(self):
        incentive1=[]
        incentive2=[]
        for i in range(self.p0.action_len):
            g1=np.eye(self.p0.action_len)[i].reshape(1,self.p0.action_len)@self.p0.utility@self.p1.action-self.p0.action.T@self.p0.utility@self.p1.action
            incentive1.append(g1[0][0])
        for j in range(self.p1.action_len):
            g2=self.p0.action.T@self.p1.utility@np.eye(self.p1.action_len)[j].reshape(self.p1.action_len,1)-self.p0.action.T@self.p1.utility@self.p1.action
            incentive2.append(g2[0][0])
        g=max(max(incentive1),max(incentive2))
        return g

    def calculateSigma_1(self):
        g1=Nash.getIncentive(self)
        beta=0.445041
        if g1>=0 and g1<=1/3:
            sigma_1=0
        elif g1>1/3 and g1<= beta:
            sigma_1=(1-g1)*(-1+(1+1/(1-2*g1)-1/g1)**0.5)
        else:
            sigma_1=1
        return sigma_1

    def getAction1(self):
        sigma_1=Nash.calculateSigma_1(self)
        r1=np.eye(self.p0.action_len)[np.argmax(np.eye(self.p0.action_len)@self.p0.utility@self.p1.action)].reshape(self.p0.action_len,1)
        b2=np.eye(self.p1.action_len)[np.argmax(((1-sigma_1)*self.p0.action+sigma_1*r1).T@self.p1.utility@np.eye(self.p1.action_len))].reshape(self.p1.action_len,1)
        h2=self.p0.action.T@self.p1.utility@b2-self.p0.action.T@self.p1.utility@self.p1.action
        return r1,b2,h2
    
    def calculateSigma_2(self):
        sigma_1=Nash.calculateSigma_1(self)
        g1=Nash.getIncentive(self)
        h2=np.array(Nash.getAction1(self))[2]
        beta=0.445041
        if g1>=0 and g1<=1/3:
            sigma_2=0
        elif g1>1/3 and g1<= beta:
            sigma_2=max(0,(sigma_1-g1+(1-sigma_1)*h2)/(1+sigma_1-g1))
        else:
            sigma_2=(1-g1)/(2-g1)
        return sigma_2

    def getAction2(self):
        sigma_1=Nash.calculateSigma_1(self)
        sigma_2=Nash.calculateSigma_2(self)
        r1=Nash.getAction1(self)[0]
        b2=Nash.getAction1(self)[1]
        act_p0=(1-sigma_1)*self.p0.action+sigma_1*r1
        act_p1=(1-sigma_2)*self.p1.action+sigma_2*b2
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
    x_star=
    y_star=
    p0 = Player(2, u0, 0,x_star)
    p1 = Player(2, u1, 1,y_star)
    nash = Nash(p0, p1)
    nash.getAction2()
