
def show_strategy(headline, players, strategies):
    print("")
    print(headline)
    for player_id in range(players):
        print(player_id, ":")
        print(strategies[player_id])

def show_eps(headline, maxeps, players, eps):
    print("")
    print(headline)
    print("maxeps:", maxeps)
    for player_id in range(players):
        print(player_id, ":", eps[player_id])
