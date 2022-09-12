import numpy as np

# return
# ANE: maximal eps and that for each player with game utility matrix
# WSNE: maximal well-supported eps and that for each player with game utility matrix

def _epsNE_with_utility(players, actions, strategies, utility, NON_ZERO=1e-10):

    eps = []
    epsWS = []

    for player_id in range(players):
        n_util = np.moveaxis(utility[..., player_id], player_id, -1)
        for other_player_id in range(players):
            if other_player_id != player_id:
                n_util = strategies[other_player_id] @ n_util
        eps.append(float(np.max(n_util) - strategies[player_id] @ n_util))
        epsWS.append(float(np.max(n_util) - np.min(np.where(strategies[player_id] > NON_ZERO, n_util, np.ones_like(n_util)))))

    ret = []
    ret.append((max(eps), eps.copy()))
    ret.append((max(epsWS), epsWS.copy()))
    return ret

# parse utility matrix from interface game.play with DFS

def _dfs_parse_utility(players, actions, play, cur_action, utility, player_id, samples):

    if player_id == players:
        tmp = np.array([0.] * players)
        for sample_id in range(samples):
            tmp += play(cur_action)
        utility.append(tmp / samples)
        return

    for action_id in range(actions[player_id]):
        cur_action[player_id] = action_id
        _dfs_parse_utility(players, actions, play, cur_action, utility, player_id + 1, samples)

# parse utility matrix from game

def sample_utility(game, samples = 1):
    utility = []
    players = game.players
    actions = game.getActionSpace()
    cur_action = [0] * players
    _dfs_parse_utility(players, actions, game.play, cur_action, utility, 0, samples)
    utility = np.array(utility).reshape(tuple(actions) + (players,))
    return utility

# return maximal eps and eps for each player with game utility matrix

def epsNE_with_sample(game, strategies, samples = 1):

    players = game.players
    actions = game.getActionSpace()
    utility = sample_utility(game, samples)
    return _epsNE_with_utility(players, actions, strategies, utility)

