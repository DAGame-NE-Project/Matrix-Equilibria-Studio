from runner import Runner
from algo import Solver
from env import Game
from env.gen import GameGenerator
import argparse
import yaml
import sys

def run(args):

    timeInterval = 100000

    currentGenerator = GameGenerator[args.generator](args)
    currentGame = Game[args.game](currentGenerator)
    currentSolver = Solver[args.solver](args)
    currentRunner = Runner[args.runner](currentGame, currentSolver, args)
    currentRunner.episode(timeInterval)

    print("last_strategy:", currentRunner.strategy[-1])
    print("last_info:", currentRunner.info[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver")
    parser.add_argument("--game")
    parser.add_argument("--generator")
    parser.add_argument("--runner")
    parser.add_argument("--players")
    parser.add_argument("--actionspace")
    args = parser.parse_args()

    if args.players is None:
        args.players = 2
    else:
        args.players = int(args.players)

    if args.actionspace is not None:
        args.actionspace = eval(args.actionspace)
        assert(len(args.actionspace) == args.players)
    else:
        args.actionspace = [2] * args.players

    run(args)
