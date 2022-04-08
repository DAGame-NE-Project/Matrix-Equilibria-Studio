import numpy as np


class Player:
    def __init__(self, experts):
        self.sum_p = np.full(3, 0.)
        # total game number
        self.games_played = 0
        # number of experts
        self.n = len(experts)
        # experts list
        self.experts = experts
        # cumulative expected reward for our Regret Matching algorithm
        self.expected_reward = 0.
        # cumulative expected rewards for experts
        self.experts_rewards = np.zeros(self.n)
        # cumulative regrets towards experts
        self.regrets = np.zeros(self.n)
        # probability disribution over experts to draw decision from
        self.p = np.full(self.n, 1. / self.n)

    def move(self):
        action = np.random.choice(self.experts, 1,  p=self.p)
        return action[0]
    
    def _update_rule(self, rewards_vector):
        self.expected_reward += np.dot(self.p, rewards_vector)
        self.experts_rewards += rewards_vector
        self.regrets = self.experts_rewards - self.expected_reward
        self._update_p()
        
    def _update_p(self):
        sum_w = np.sum([self._w(i) for i in np.arange(self.n)])
        if sum_w <= 0:
            self.p = np.full(self.n, 1. / self.n)
        else:
            self.p = np.asarray(
                [self._w(i) / sum_w for i in np.arange(self.n)]
            )

    def _w(self, i):
        return max(0, self.regrets[i])

    def learn_from(self, opponent_move):
        reward_vector = REWARD_VECTORS[opponent_move]
        self._update_rule(reward_vector)
        self.games_played += 1
        self.sum_p += self.p

    def current_best_response(self):
        return np.round(self.sum_p / self.games_played, 4)

    def eps(self):
        return np.max(self.regrets / self.games_played)
 
class Nash(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def getActionSpace(self, iterations):
        """
        求解纳什均衡
        :param  iterations: 迭代次数
        """
        for i in range(0, iterations):
            p0_move = self.p0.move()
            p1_move = self.p1.move()
            self.p0.learn_from(p0_move)
            self.p1.learn_from(p1_move)

        _2e = np.round(2 * np.max([self.p0.eps(), self.p1.eps()]), 3)
        p0_ne = self.p0.current_best_response()
        p1_ne = self.p1.current_best_response()
        print("{0} - nash equilibrium for RPS: {1}, {2}".format(_2e, p0_ne, p1_ne))

        
"""
if __name__ == '__main__':
    ACTION = ['ROCK', 'PAPER', 'SCISSORS']

    REWARD_VECTORS = {
    'ROCK':     np.asarray([0, 1, -1]),  # opponent playing ROCK
    'PAPER':    np.asarray([-1, 0, 1]),  # opponent playing PAPER
    'SCISSORS': np.asarray([1, -1, 0]),  # opponent playing PAPER
}
    a = Player(ACTION)
    b = Player(ACTION)
    t = 10000
    nash=Nash(a,b)
    nash.getActionSpace(t)
"""
