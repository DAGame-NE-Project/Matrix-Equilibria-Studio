from .iterative_solver import IterativeSolver
from .direct_solver import DirectSolver

from .BBM_36 import Player as BBM_36
from .DMP_38 import Player as DMP_38
from .DMP_50 import Player as DMP_50
from .KPS_75 import Player as KPS_75
from .lemke_howson import LemkeHowson
from .support_enumeration import SupportEnumeration

from .Hedge import Hedge
from .MW import MW as MWU
from .RM import Player as RM
from .fictitious_play import Player as FictitiousPlay
from .random_play import RandomPlay
from .replicator_dynamics import ReplicatorDynamics

Solver = dict()

Solver['bbm_36'] = BBM_36
Solver['dmp_38'] = DMP_38
Solver['dmp_50'] = DMP_50
Solver['kps_75'] = KPS_75
Solver['lh'] = LemkeHowson
Solver['se'] = SupportEnumeration

Solver['hedge'] = Hedge
Solver['mwu'] = MWU
Solver['rm'] = RM
Solver['fp'] = FictitiousPlay
Solver['random'] = RandomPlay
Solver['rp'] = ReplicatorDynamics
