from .direct_solver import DirectSolver
import numpy as np

class Player(DirectSolver):

    def __init__(self, args):
        super(Player, self).__init__(args)

    def solve(self, game, utility):

        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "KPS_75 only works for 2-player games!"
        ret = []

        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))

        for player_id in range(players):
            max_util_idx = np.unravel_index(np.argmax(utility[player_id]), actions)
            for enum_id in range(players):
                ret[enum_id][max_util_idx[enum_id]] += 1.0

        for player_id in range(players):
            ret[player_id] /= players

        info = {
            'solver': "KPS_75",
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
        }

        return ret, info

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
#        self.action = np.random.random(self.action_len)
#        self.id = id
#
#
#class Nash(object):
#    def __init__(self, p0, p1):
#        self.p0 = p0
#        self.p1 = p1
#
#    def getActionSpace(self):
#
#        p0_action=np.array([1.0,0.0])
#        p1_action=np.array([1.0,0.0])
#
#        while(True):
#            
#            
#            c = np.matmul(p0_action.T,self.p1.utility)
#            r = np.matmul(self.p0.utility,p1_action)
#            mix1 = p0_action.T@self.p1.utility@p1_action
#            mix2 = p0_action.T@self.p0.utility@p1_action
#            flag = 1
#            for i in range(len(c)):
#                if(mix1<c[i] or mix2<r[i]):
#                    flag = 0
#            
#            if flag == 1:
#                break
#            r_index = np.argmin(r)
#            c_index = np.argmax(c)
#
#            for i in range(p0_action.shape[0]):
#
#                if i!=r_index:
#                    p0_action[i] = p0_action[i]/2
#                   
#                else:
#                    p0_action[i] = p0_action[i]/2 + 1/2
#
#            for i in range(p1_action.shape[0]):
#                if i!=c_index:
#                    p1_action[i] = p1_action[i]/2
#                else:
#                    p1_action[i] = p1_action[i]/2 + 1/2
#                    
#
#        print('p0', p0_action)
#        print('p1', p1_action)
#
#
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
