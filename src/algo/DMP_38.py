from .direct_solver import DirectSolver
from util import solve_lp
import numpy as np
import random

class Player(DirectSolver):

    def __init__(self, args):
        super(Player, self).__init__(args)
        self.one_over_epsilon = args.one_over_epsilon if hasattr(args, 'one_over_epsilon') else 1000

    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "DMP_38 is only works for 2-player games!"
        ret = self._enumerate_solution(actions, utility)

        info = {
            'solver': "DMP_38",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
        }

        return ret, info


    def _enumerate_solution(self, actions, utility):

        def generate_all_strategies(dims, one_over_epsilon):
            tmp = np.zeros(dims + 1, dtype=int)
            tmp[0] = one_over_epsilon
            while tmp[dims] != one_over_epsilon:
                non_zero_idx = 0
                while tmp[non_zero_idx] == 0:
                    non_zero_idx += 1
                k = tmp[non_zero_idx]
                tmp[non_zero_idx] = 0
                tmp[non_zero_idx] += 1
                tmp[0] = k - 1
                if tmp[dims] != one_over_epsilon:
                    yield tmp[:dims].astype(float) / (one_over_epsilon - tmp[dims])

        def lp_solution(vr, vc, alpha, beta, eps_over_two):
            n = actions[0]
            m = actions[1]
            A_ge = []
            b_ge = []
            A_le = []
            b_le = []
            A_eq = []
            b_eq = []

            A_ge.append(np.concatenate([np.zeros(n), alpha @ utility[0]], axis=0).reshape(1, n + m))
            b_ge.append(np.array([vr - 3 * eps_over_two]))

            A_le.append(np.concatenate([np.zeros((n,n)), utility[0]], axis=1))
            b_le.append(np.ones(n) * (vr + eps_over_two))

            A_ge.append(np.concatenate([beta @ utility[0].T, np.zeros(m)], axis=0).reshape((1, n + m)))
            b_ge.append(np.array([vr - 3 * eps_over_two]))

            A_ge.append(np.concatenate([np.zeros(n), alpha @ utility[1]], axis=0).reshape(1, n + m))
            b_ge.append(np.array([vc - 3 * eps_over_two]))

            A_le.append(np.concatenate([utility[1].T, np.zeros((m, m))], axis=1))
            b_le.append(np.ones(m) * (vc + eps_over_two))

            A_ge.append(np.concatenate([beta @ utility[1].T, np.zeros(m)], axis=0).reshape((1, n + m)))
            b_ge.append(np.array([vc - 3 * eps_over_two]))

            A_eq.append(np.concatenate([np.ones(n), np.zeros(m)], axis=0).reshape((1, n + m)))
            A_eq.append(np.concatenate([np.ones(m), np.zeros(n)], axis=0).reshape((1, n + m)))
            b_eq = [np.ones(2)]

            A_le = np.concatenate(A_le, axis=0)
            b_le = np.concatenate(b_le, axis=0)
            A_eq = np.concatenate(A_eq, axis=0)
            b_eq = np.concatenate(b_eq, axis=0)
            A_ge = np.concatenate(A_ge, axis=0)
            b_ge = np.concatenate(b_ge, axis=0)

            c = np.ones(n + m)
            sol = solve_lp(A_le = A_le, b_le = b_le, A_eq = A_eq, b_eq = b_eq, A_ge =A_ge, b_ge = b_ge, c = c)
            return sol.x[:n].copy(), sol.x[n:].copy()

        eps_over_two = 0.5 / self.one_over_epsilon
        for kr in range(self.one_over_epsilon + 1):
            vr = 1. / self.one_over_epsilon * kr
            for kc in range(self.one_over_epsilon + 1):
                vc = 1. / self.one_over_epsilon * kc
                for alpha in generate_all_strategies(actions[0], 4 * self.one_over_epsilon ** 2):
                    for beta in generate_all_strategies(actions[1], 4 * self.one_over_epsilon ** 2):
                        if alpha @ utility[0] @ beta >= vr - 3 * eps_over_two and
                            alpha @ utility[1] @ beta >= vc - 3 * eps_over_two:
                                x, y = lp_solution(vr, vc, alpha, beta, eps_over_two)
                                if max(vr, vc) * 3 >= one_over_epsilon:
                                    delta = 1.5 - (0.5 * self.one_over_epsilon / max(vr, vc))
                                    return [delta * alpha + (1 - delta) * x, delta * beta + (1 - delta) * y]
                                else:
                                    return [x, y]


    def reset(self):
        pass

#    def __init__(self, action_len, utility, id):
#        """
#        :param action_len: 策略个数
#        :param utility:  收益矩阵
#        :param id
#        """
#        self.utility = utility
#        self.action_len = action_len
#        self.action = [0,0]
#        self.id = id
#
#
#class Nash(object):
#    def __init__(self, p0, p1):
#        self.p0 = p0
#        self.p1 = p1
#
#    def getActionSpace(self):        
#        epsilon = 0.1
#   
#        alpha_list = []
#        beta_list = []
#        r_list = []
#        c_list = []
#        
#        for m in range(int(1/epsilon)+1):
#            for n in range(int(1/epsilon)+1):
#                v_r = epsilon*m
#                v_c = epsilon*n
#                for i in range(int(1/epsilon)+1):
#                    flag = 0
#                    for j in range(int(1/epsilon)+1):
#                        alpha = np.array([i*epsilon,1-i*epsilon])
#                        beta = np.array([j*epsilon,1-j*epsilon])
#                        r_pay = alpha@self.p0.utility@beta
#                        c_pay = alpha@self.p1.utility@beta
#
#                        if r_pay >= v_r-1.5*epsilon and c_pay >= v_c-1.5*epsilon:
#                            alpha_list.append(alpha)
#                            beta_list.append(beta)
#                            r_list.append(v_r)
#                            c_list.append(v_c)
#                            
#
#        rv = 0
#        cv = 0
#        
#        for v_r,v_c,alpha,beta in zip(r_list,c_list,alpha_list,beta_list):
#
#            flag = 0
#            for i in range(int(1/epsilon)+1):
#                for j in range(int(1/epsilon)+1):
#                    x = np.array([i*epsilon,1-i*epsilon])
#                    y = np.array([j*epsilon,1-j*epsilon])
#                    r_1 = (alpha@self.p0.utility@y >= v_r - 1.5*epsilon)
#                    r_2 = (np.array([1,0])@self.p0.utility@y <= v_r + 0.5*epsilon and np.array([0,1])@self.p0.utility@y <= v_r + 0.5*epsilon)
#                    r_3 = (x@self.p0.utility@beta >= v_r - 1.5*epsilon)
#                    c_1 = (alpha@self.p1.utility@y >= v_c - 1.5*epsilon)
#                    c_2 = (x@self.p1.utility@np.array([1,0]) <= v_c + 0.5*epsilon and x@self.p1.utility@np.array([0,1]) <= v_c + 0.5*epsilon)
#                    c_3 = (x@self.p1.utility@beta >= v_c - 1.5*epsilon)
#
#                    if(r_1 and r_2 and r_3 and c_1 and c_2 and c_3):
#                        flag = 1
#                        break
#                if(flag):
#                    break
#            if(flag):
#                v_max = max(v_c,v_r)
#                if v_max >= 1/3:
#                    derta = 3/2-1/(2*v_max)
#                    x = derta*alpha + (1-derta)*x
#                    y = derta*beta + (1-derta)*y
#                else:
#                    pass
#
#                r_tmp = x@self.p0.utility@y
#                c_tmp = x@self.p1.utility@y
#                
#                if r_tmp >= rv and c_tmp >= cv:
#                    rv = r_tmp
#                    self.p0.action = x
#                    cv = c_tmp
#                    self.p1.action = y
#            
#
#        
#        print('p0',self.p0.action)
#        print('p1',self.p1.action)
#
#if __name__ == '__main__':
#
#    u0 = np.array(
#        [[1, 0],[0, 1]]
#    )
#    u1 = np.array(
#        [[0, 1],[1, 0]]
#    )
#    p0 = Player(2, u0, 0)
#    p1 = Player(2, u1, 1)
#    nash = Nash(p0, p1)
#    nash.getActionSpace()
