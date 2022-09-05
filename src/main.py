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
from util import epsNE_with_sample, show_eps, show_strategy

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_PATH = os.path.join(PROJECT_PATH, "results")

def run_episode(args, timeInterval = 100000, record_info = None):

    # timeInterval = 100000
    eval_samples = args.eval_samples if hasattr(args, 'eval_samples') else 1

    currentGenerator = GameGenerator[args.generator](args)
    currentGame = Game[args.game](currentGenerator)
    currentSolver = Solver[args.solver](args)
    currentRunner = Runner[args.runner](currentGame, currentSolver, args)
    if record_info is None:
        currentRunner.episode(timeInterval)
    else:
        record_avg_eps = []
        record_last_eps = []
        record_avg_ws_eps = []
        record_last_ws_eps = []
        for time_stamp in tqdm(range(timeInterval)):
            currentRunner.step()
            if 'overall_policy' in currentRunner.current_info:
                strategies = list(map(lambda x: np.array(x) / (time_stamp + 1), currentRunner.current_info['overall_policy']))
                ret = epsNE_with_sample(currentGame, strategies, 1)
                maxeps, eps = ret[0]
                record_avg_eps.append((maxeps, eps))
                maxwseps, wseps = ret[1]
                record_avg_ws_eps.append((maxwseps, wseps))
            if True:
                strategies = currentRunner.strategy[-1]
                ret = epsNE_with_sample(currentGame, strategies, 1)
                maxeps, eps = ret[0]
                record_last_eps.append((maxeps, eps))
                maxwseps, wseps = ret[1]
                record_last_ws_eps.append((maxwseps, wseps))
        output_dict = dict({})
        output_dict['avg_eps'] = record_avg_eps
        output_dict['last_eps'] = record_last_eps
        output_dict['avg_ws_eps'] = record_avg_ws_eps
        output_dict['last_ws_eps'] = record_last_ws_eps
        import json
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)
        SUBDIR_PATH = os.path.join(RESULT_PATH, args.solver)
        if not os.path.exists(SUBDIR_PATH):
            os.mkdir(SUBDIR_PATH)
        with open(os.path.join(SUBDIR_PATH, record_info['file_name'] + ".json"), 'w') as fp:
            json.dump(output_dict, fp, separators=(",\n",":\n"))

    # if need to calc average policy, please use 'overall_policy' counting the strategies in info

    if 'overall_policy' in currentRunner.current_info:
        strategies = list(map(lambda x: np.array(x) / timeInterval, currentRunner.current_info['overall_policy']))
        show_strategy("Average Strategy", currentGame.players, strategies)
        ret = epsNE_with_sample(currentGame, strategies, 1)
        maxeps, eps = ret[0]
        show_eps("EPS Info", maxeps, currentGame.players, eps)
        maxwseps, wseps = ret[1]
        show_eps("WSEPS Info", maxwseps, currentGame.players, wseps)

    # calc last_strategy (last iteration)

    if True:
        strategies = currentRunner.strategy[-1]
        show_strategy("Last Iteration", currentGame.players, strategies)
        ret = epsNE_with_sample(currentGame, strategies, 1)
        maxeps, eps = ret[0]
        show_eps("EPS Info", maxeps, currentGame.players, eps)
        maxwseps, wseps = ret[1]
        show_eps("WSEPS Info", maxwseps, currentGame.players, wseps)

    # print("last_strategy:", currentRunner.strategy[-1])
    print("last_info:", currentRunner.info[-1])

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


    file_prefix = args.solver
    if args.type is not None:
        file_prefix = file_prefix +  "_" + args.type
    for test_num in range(args.totaltestnum):
        file_name = file_prefix + str(test_num)
        record_info = {
            'file_name': file_prefix + str(test_num)
            }
        print(file_name, "START!!!({}/{})".format(test_num + 1, args.totaltestnum))
        run_episode(args, record_info = record_info)
        print(file_name, "FINISH!!!({}/{})".format(test_num + 1, args.totaltestnum))
