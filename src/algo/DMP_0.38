import numpy as np
import random

class Player(object):
    def __init__(self, action_len, utility, id):
        """
        :param action_len: 策略个数
        :param utility:  收益矩阵
        :param id
        """
        self.utility = utility
        self.action_len = action_len
        self.action = [0,0]
        self.id = id


class Nash(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def getActionSpace(self):        
        epsilon = 0.1
   
        alpha_list = []
        beta_list = []
        r_list = []
        c_list = []
        
        for m in range(int(1/epsilon)+1):
            for n in range(int(1/epsilon)+1):
                v_r = epsilon*m
                v_c = epsilon*n
                for i in range(int(1/epsilon)+1):
                    flag = 0
                    for j in range(int(1/epsilon)+1):
                        alpha = np.array([i*epsilon,1-i*epsilon])
                        beta = np.array([j*epsilon,1-j*epsilon])
                        r_pay = alpha@self.p0.utility@beta
                        c_pay = alpha@self.p1.utility@beta

                        if r_pay >= v_r-1.5*epsilon and c_pay >= v_c-1.5*epsilon:
                            alpha_list.append(alpha)
                            beta_list.append(beta)
                            r_list.append(v_r)
                            c_list.append(v_c)
                            

        rv = 0
        cv = 0
        
        for v_r,v_c,alpha,beta in zip(r_list,c_list,alpha_list,beta_list):

            flag = 0
            for i in range(int(1/epsilon)+1):
                for j in range(int(1/epsilon)+1):
                    x = np.array([i*epsilon,1-i*epsilon])
                    y = np.array([j*epsilon,1-j*epsilon])
                    r_1 = (alpha@self.p0.utility@y >= v_r - 1.5*epsilon)
                    r_2 = (np.array([1,0])@self.p0.utility@y <= v_r + 0.5*epsilon and np.array([0,1])@self.p0.utility@y <= v_r + 0.5*epsilon)
                    r_3 = (x@self.p0.utility@beta >= v_r - 1.5*epsilon)
                    c_1 = (alpha@self.p1.utility@y >= v_c - 1.5*epsilon)
                    c_2 = (x@self.p1.utility@np.array([1,0]) <= v_c + 0.5*epsilon and x@self.p1.utility@np.array([0,1]) <= v_c + 0.5*epsilon)
                    c_3 = (x@self.p1.utility@beta >= v_c - 1.5*epsilon)

                    if(r_1 and r_2 and r_3 and c_1 and c_2 and c_3):
                        flag = 1
                        break
                if(flag):
                    break
            if(flag):
                v_max = max(v_c,v_r)
                if v_max >= 1/3:
                    derta = 3/2-1/(2*v_max)
                    x = derta*alpha + (1-derta)*x
                    y = derta*beta + (1-derta)*y
                else:
                    pass

                r_tmp = x@self.p0.utility@y
                c_tmp = x@self.p1.utility@y
                
                if r_tmp >= rv and c_tmp >= cv:
                    rv = r_tmp
                    self.p0.action = x
                    cv = c_tmp
                    self.p1.action = y
            

        
        print('p0',self.p0.action)
        print('p1',self.p1.action)

if __name__ == '__main__':

    u0 = np.array(
        [[1, 0],[0, 1]]
    )
    u1 = np.array(
        [[0, 1],[1, 0]]
    )
    p0 = Player(2, u0, 0)
    p1 = Player(2, u1, 1)
    nash = Nash(p0, p1)
    nash.getActionSpace()
