from .iterative_solver import IterativeSolver
from .direct_solver import DirectSolver

from .BBM_36 import Player as BBM_36
from .DMP_38 import Player as DMP_38
from .DMP_50 import Player as DMP_50
from .KPS_75 import Player as KPS_75

from .Hedge import Hedge
from .fictitious_play import Player as FictitiousPlay
from .MW import MW as MWU
from .random_play import RandomPlay
from .RM import Player as RM

Solver = dict()

Solver['bbm_36'] = BBM_36
Solver['dmp_38'] = DMP_38
Solver['dmp_50'] = DMP_50
Solver['kps_75'] = KPS_38
Solver['random'] = RandomPlay
Solver['fp'] = FictitiousPlay
Solver['mwu'] = MWU
Solver['hedge'] = Hedge
Solver['rm'] = RM
