import numpy as np


class Player(object):

    def __init__(self, args):

        self.args = args
        if hasattr(args, 'epsilon'):
            self.epsilon = float(args.epsilon)
        else:
            self.epsilon = 1e-20

    def solve(self, game, info):

        actions = game.getActionSpace()
        players = game.players
        if info is None or info['solver'] != "FictitiousPlay":
            info = {
                'solver': "FictitiousPlay",
                'agents_history_reward': [[0.0 for action_id in range(actions[player_id])] for player_id in range(players)],
                'overall_policy': [[0.0 for action_id in range(actions[player_id])] for player_id in range(players)],
            }

        ret = tuple()
        for player_id in range(players):
            maximal_reward = max(info['agents_history_reward'][player_id])
            choices = []
            for action_id in range(actions[player_id]):
                if info['agents_history_reward'][player_id][action_id] + self.epsilon >= maximal_reward:
                    choices.append(action_id)
            ret = ret + tuple([int(np.random.choice(choices))])
        self._update_info(game, ret, info)

        tmp_ret = ret
        ret = [np.zeros(actions[player_id]) for player_id in range(players)]
        for player_id in range(players):
            ret[player_id][tmp_ret[player_id]] = 1.
        # strategy action info
        return ret, tmp_ret, info

    def _update_info(self, game, ret, info):

        actions = game.getActionSpace()
        for player_id in range(game.players):
            tmp_ret = list(ret)
            for action_id in range(actions[player_id]):
                tmp_ret[player_id] = action_id
                info['agents_history_reward'][player_id][action_id] += game.playID(tmp_ret, player_id)
            info['overall_policy'][player_id][ret[player_id]] += 1

    def reset(self):
        pass

#    def __init__(self, action_len, utility, id):
#        """
#        :param action_len: 策略个数
#        :param utility:  收益矩阵
#        :param id: 玩家0写0 玩家1写1
#        """
#        self.utility = utility
#        self.action_len = action_len
#        self.action = np.random.random(self.action_len)
#        self.history = np.zeros(self.action_len)
#        self.id = id
#
#    def change_action(self, op_pro):
#        """
#        根据传入的对手历史策略，选择自己的最优策略，并改变自己的策略
#        :param op_pro: 对手策略
#        """
#        earn = op_pro * self.utility
#        earn_sum = np.sum(earn, axis=1 - self.id)
#        best_choice = np.argmax(earn_sum)
#        self.history[best_choice] += 1
#        self.action = self.history / np.sum(self.history)
#
#    def get_action(self):
#        """
#        return: 返回自己本轮策略
#        """
#        if self.id == 0:
#            return np.reshape(self.action, (self.action_len, 1))
#        else:
#            return self.action
#
#
#class Nash(object):
#    def __init__(self, p0, p1):
#        self.p0 = p0
#        self.p1 = p1
#
#    def getActionSpace(self, iterations):
#        """
#        求解
#        :param  iterations: 迭代次数
#        """
#        for i in range(iterations):
#            self.p0.change_action(self.p1.get_action())
#            self.p1.change_action(self.p0.get_action())
#        print('p0', self.p0.get_action())
#        print('p1', self.p1.get_action())
#
#
#
#if __name__ == '__main__':
#    
#    # 囚徒困境示例
#    #     P0╲P1    坦白    抵赖
#    #      坦白   -4，-4   0，-5
#    #      抵赖   -5， 0  -1，-1
#    
#    u0 = np.array(
#        [[-4, 0],
#         [-5, -1]]
#    )
#    u1 = np.array(
#        [[-4, -5],
#         [0, -1]]
#    )
#    p0 = Player(2, u0, 0)
#    p1 = Player(2, u1, 1)
#    """
#    # Player3:
#    #     P0╲P1    石头    剪刀    布
#    #      石头    0, 0   1,-1  -1, 1
#    #      剪刀   -1, 1   0, 0   1,-1
#    #       布     1,-1  -1, 1   0, 0
#
#    u0 = np.array(
#        [[0, 1, -1],
#         [-1, 0, 1],
#         [1, -1, 0]]
#    )
#    u1 = np.array(
#        [[0, -1, 1],
#         [1, 0, -1],
#         [-1, 1, 0]]
#    )
#    
#    p0 = Player(3, u0, 0)
#    p1 = Player(3, u1, 1)
#    """
#    nash = Nash(p0, p1)
#    nash.getActionSpace(1000)
