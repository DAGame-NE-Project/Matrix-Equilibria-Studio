import numpy as np


class Hedge(object):

    def __init__(self, args):

        self.args = args
        if hasattr(args, 'epsilon'):
            self.epsilon = float(args.epsilon)
        else:
            self.epsilon = 0

    def solve(self, game, info):

        actions = game.getActionSpace()
        players = game.players
        if info is None or info['solver'] != "Hedge":
            info = {
                'solver': "Hedge",
                'agents_history_log_weights': [[0.0 for action_id in range(actions[player_id])] for player_id in range(players)],
                'overall_policy': [[0.0 for action_id in range(actions[player_id])] for player_id in range(players)],
            }

        ret = []
        tmp_ret = tuple()

        for player_id in range(players):
            policy = np.array(info['agents_history_log_weights'][player_id])
            policy = np.exp(policy - np.max(policy))
            policy = policy / np.sum(policy)
            ret.append(policy)
            tmp_ret = tmp_ret + tuple([int(np.random.choice(np.array(range(actions[player_id])), p = policy))])

        self._update_info(game, tmp_ret, info)

        for player_id in range(players):
            info['overall_policy'][player_id] += ret[player_id]

        return ret, tmp_ret, info

    def _update_info(self, game, ret, info):

        actions = game.getActionSpace()
        for player_id in range(game.players):
            tmp_ret = list(ret)
            for action_id in range(actions[player_id]):
                tmp_ret[player_id] = action_id
                u = game.playID(tmp_ret, player_id)
                inc = np.log(1 + self.epsilon) * u
                info['agents_history_log_weights'][player_id][action_id] += inc

    def reset(self):
        pass


#    def __init__(self,n=10,T=10,a=1):
#        """
#        Args:
#            n (int): number of experts present
#            T (int): number of time steps taken
#            a (float): value that we can use to scale our epsilon
#            epsilon (float): the theoretical episilon, but which can be customized by a
#        """
#        self.n = n
#        self.T = T
#        self.a = a
#        self.weights = np.ones(n)/n
#        self.epsilon = np.sqrt(np.log(n) / T) * a
#
#    def weight(self,expert_predictions,actual_value):
#        """
#        Will recalculate the weights on each of the experts based off of the expert predictions and the actual value.
#        This can only be done after the actual value is known.
#
#        Args:
#            expert_predictions (np.array) (pred.float): np.array with the expert predictions
#            actual_values (float): float of the actual value here
#
#        Returns:
#            Nothing
#        """
#
#        #Calculating losses
#        losses = self.se(expert_predictions,actual_value)
#
#        #Apply weights
#        reweighting = np.exp(-self.epsilon*losses)
#        self.weights*=reweighting
#        self.weights/=np.sum(self.weights)
#
#    def predict(self,expert_predictions):
#        """
#        Weights the expert predictions into a single prediction based on the weights that have been calculated by the
#        hedge algorithm
#
#        Args:
#            expert predictions (np.array) (pred.float): np.array with the expert predictions
#
#        Returns:
#            a value for prediction based on the inputs of the experts and their respective weights.
#        """
#
#        return np.dot(self.weights,expert_predictions)
#
#    def fit_predict(self,expert_predictions,actual_values):
#        """
#        Will perform weighting at each time step of historical data in order to fit to the data and also perform online
#        predictions. Shows the performance of the algorithm and can be used to check performance.
#
#        Args:
#            expert_predictions (np.array) (pred.float,time.float): np.array with the expert predictions across time
#            actual_values (np.array) (time.float): np.array with the actual value across time.
#
#        Returns:
#            (weights (np.array), predictions (np.array))
#        """
#        weights = []
#        predictions = []
#        for i in range(len(actual_values)):
#            weights.append(hedge.weights.copy())
#            predictions.append(hedge.predict(expert_predictions[:,i]))
#            hedge.re_weight(expert_predictions[:,i],actual_values[i])
#
#        return (np.array(weights), np.array(predictions))
#
#    def se(actual,expected):
#        """
#        Will return the squared error between the two arguments
#        """
#        return np.power(actual-expected,2)
#    
#    def run():
#        """
#        run to update the weights
#        
#        Return:
#            updated weights
#        """
#        for i in range(self.T):
#            weights.append(self.weights.copy())
#            self.weight(expert_predictions[:,i],actual_values[i])
#        
#        return self.weight
#    
#if __name__ == '__main__':
#
#    actual_values = np.random.randint(0,10,size=[20])
#
#    # Fixed Standard Deviations for each expert
#    stds = [1,0.1,0.03,0.5,1.3]
#    expert_predictions = np.array([actual_values+np.random.randn(actual_values.size)*std for std in stds])
#
#    hedge = onlineHedge(n=5,T=20,a=1)
#    #run to update weights
#    
#    hedge.run()
#    predictions = []
#    predictions.append(hedge.predict(expert_predictions[:]))
