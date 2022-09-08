from .iterative_solver import IterativeSolver
from .direct_solver import DirectSolver
from .Hedge import Hedge
from .fictitious_play import Player as FictitiousPlay
from .MW import MW as MWU
from .random_play import RandomPlay
from .RM import Player as RM
Solver = dict()

Solver['random'] = RandomPlay
Solver['fp'] = FictitiousPlay
Solver['mwu'] = MWU
Solver['hedge'] = Hedge
Solver['rm'] = RM
