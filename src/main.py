from runner import *

def run(args):
    currentGenerator = GameGenerator[args['generator']](args)
    currentGame = Game[args['game']](currentGenerator)
    currentSolver = Solver[args['solver']]
    currentRunner = Runner[args['runner']](currentGame, currentSolver)
    currentRunner.episode()
    print("last_strategy:", currentRunner.strategy[-1])
    print("last_info:", currentRunner.info[-1])


if __name__ == "__main__":
    run(args)
