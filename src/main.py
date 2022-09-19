import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
import yaml

from algo import Solver
from env import Game
from env.gen import GameGenerator
from runner import Runner
from util import update_args, ALGO_CONF_PATH, CONF_PATH, ENV_CONF_PATH, PROJECT_PATH


def run_episode(args):

    currentGenerator = GameGenerator[args.generator](args)
    currentGame = Game[args.game](currentGenerator)
    currentSolver = Solver[args.solver](args)
    currentRunner = Runner[args.runner](currentGame, currentSolver, args)
    currentRunner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="filename of yaml file in conf/algo, e.g. fp for fp.yaml")
    parser.add_argument("--env", help="filename of yaml file in conf/env, e.g. zerosum for zerosum.yaml")
    parser.add_argument("--resultpath", help="relative path for storing results, default is results/$env_$algo")
    parser.add_argument("-t", "--totaltestnum", type=int, default=-1, help="total number of testnum, default is in conf/env/$env")
    args = parser.parse_args()

    # load env/default.yaml
    with open(os.path.join(ENV_CONF_PATH, 'default.yaml'), 'r') as fp:
        default_env_dic = yaml.safe_load(fp)

    # load env.yaml
    with open(os.path.join(ENV_CONF_PATH, args.env + '.yaml'), 'r') as fp:
        env_dic = yaml.safe_load(fp)

    # load algo/default.yaml
    with open(os.path.join(ALGO_CONF_PATH, 'default.yaml'), 'r') as fp:
        default_algo_dic = yaml.safe_load(fp)

    # load algo.yaml
    with open(os.path.join(ALGO_CONF_PATH, args.algo + '.yaml'), 'r') as fp:
        algo_dic = yaml.safe_load(fp)

    # update args
    tmp_t = args.totaltestnum

    update_args(args, default_env_dic)
    update_args(args, default_algo_dic)
    update_args(args, env_dic)
    update_args(args, algo_dic)

    if tmp_t >= 0:
        args.totaltestnum = tmp_t

    # create resultpath
    if args.resultpath is not None:
        args.resultpath = os.path.join(PROJECT_PATH, args.resultpath)
    else:
        tmppath = os.path.join(PROJECT_PATH, "results")
        if not os.path.exists(tmppath):
            os.mkdir(tmppath)
        args.resultpath = os.path.join(tmppath, args.env + "_" + args.algo)

    if not os.path.exists(args.resultpath):
        os.mkdir(args.resultpath)

    # test
    for test_num in range(args.totaltestnum):
        args.file_name = str(test_num)
        print("{}_{}_{}".format(args.env, args.algo, args.file_name), "START!!!({}/{})".format(test_num + 1, args.totaltestnum))
        run_episode(args)
        print("{}_{}_{}".format(args.env, args.algo, args.file_name), "FINISH!!!({}/{})".format(test_num + 1, args.totaltestnum))
