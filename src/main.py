import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
import yaml

from runner import Runner
from algo import Solver
from env import Game
from env.gen import GameGenerator
# from util import epsNE_with_sample, show_eps, show_strategy

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_PATH = os.path.join(PROJECT_PATH, "results")

def run_episode(args, record_info = None):

    currentGenerator = GameGenerator[args.generator](args)
    currentGame = Game[args.game](currentGenerator)
    currentSolver = Solver[args.solver](args)
    currentRunner = Runner[args.runner](currentGame, currentSolver, args)
    currentRunner.run(record_info = record_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver")
    parser.add_argument("--game")
    parser.add_argument("--generator")
    parser.add_argument("--runner")
    parser.add_argument("--players")
    parser.add_argument("--actionspace")
    parser.add_argument("--type")
    parser.add_argument("-t", "--totaltestnum", type=int, default=10)
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

    args.resultpath = RESULT_PATH


    file_prefix = args.solver
    if args.type is not None:
        file_prefix = file_prefix +  "_" + args.type
    args.dir_name = file_prefix
    for test_num in range(args.totaltestnum):
        file_name = file_prefix + "_" + str(test_num)
        record_info = {
            'file_name': file_name
            }
        print(file_name, "START!!!({}/{})".format(test_num + 1, args.totaltestnum))
        run_episode(args, record_info = record_info)
        print(file_name, "FINISH!!!({}/{})".format(test_num + 1, args.totaltestnum))
