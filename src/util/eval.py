import numpy as np

# return maximal eps and eps for each player with game utility matrix

def _epsNE_with_utility(players, actions, strategies, utility):

    eps = []

    for player_id in range(game.players):
        n_util = np.moveaxis(utility, player_id, -1)
        for other_player_id in range(game.players):
            if other_player_id != player_id:
                n_util = strategies[other_player_id] @ n_util
        eps.append(float(np.max(n_util) - strategies[player_id] @ n_util))

    return max(eps), eps

# parse utility matrix from interface game.play

def dfs_parse_utility(players, actions, play, cur_action, utility, player_id, samples):

    if player_id == players:
        tmp = np.array([0.] * players)
        for sample_id in range(samples):
            tmp += play(tmp_action)
        utility.append(tmp / samples)
        return

    for action_id in range(actions[player_id]):
        cur_action[player_id] = action_id
        dfs_parse_utility(players, actions, U, cur_action, array, player_id + 1, samples)

# return maximal eps and eps for each player with game utility matrix

def epsNE_with_sample(game, strategies, samples = 1):

    utility = []
    players = game.players
    actions = game.getActionSpace()
    cur_action = [0] * players
    dfs_parse_utility(players, actions, game.play, cur_action, utility, 0, samples)
    utility = np.array(utility).reshape(tuple(actions))
    return _epsNE_with_utility(players, actions, strategies, utility)
