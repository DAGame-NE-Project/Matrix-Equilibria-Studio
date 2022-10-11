import argparse
import numpy as np

from gen.random import RandomMatrixGenerator
from gen.zerosum import ZeroSumMatrixGenerator
from util import DATA_PATH

GEN = {}
GEN['random'] = RandomMatrixGenerator
GEN['zerosum'] = ZeroSumMatrixGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", help="generator (random / zerosum)")
    parser.add_argument("--dir", help="dirname of generated matrix in data")
    parser.add_argument("--name", help="filename of generated matrix")
    parser.add_argument("--actionspace", help="actionspace of generated matrix, e.g., '[2,2]'")
    args = parser.parse_args()
    args.DATA_PATH = DATA_PATH
    args.actionspace = eval(args.actionspace)
    GEN[args.gen](args)
