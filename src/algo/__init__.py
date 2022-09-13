from .iterative_solver import IterativeSolver
from .direct_solver import DirectSolver

from .BBM_36 import Player as BBM_36
from .CDFFJS_38 import Player as CDFFJS_38
from .CDFFJS_6528 import Player as CDFFJS_6528
from .DFM_50 import Player as DFM_50
from .DMP_38 import Player as DMP_38
from .DMP_50 import Player as DMP_50
from .FGSS_6605 import Player as FGSS_6605
from .KPS_75 import Player as KPS_75
from .KS_2_3 import Player as KS_2_3
from .lemke_howson import LemkeHowson
from .support_enumeration import SupportEnumeration
from .TS import Player as TS
from .k_uniform_search import Player as KUniformSearch

from .Hedge import Hedge
from .MW import MW as MWU
from .RM import Player as RM
from .fictitious_play import Player as FictitiousPlay
from .random_play import RandomPlay
from .replicator_dynamics import ReplicatorDynamics

Solver = dict()

Solver['bbm_36'] = BBM_36
Solver['cdffjs_38'] = CDFFJS_38
Solver['cdffjs_6528'] = CDFFJS_6528
Solver['dfm_50'] = DFM_50
Solver['dmp_38'] = DMP_38
Solver['dmp_50'] = DMP_50
Solver['fgss_6605'] = FGSS_6605
Solver['kps_75'] = KPS_75
Solver['ks_2_3'] = KS_2_3
Solver['lh'] = LemkeHowson
Solver['se'] = SupportEnumeration
Solver['ts'] = TS
Solver['uniformsearch'] = KUniformSearch

Solver['hedge'] = Hedge
Solver['mwu'] = MWU
Solver['rm'] = RM
Solver['fp'] = FictitiousPlay
Solver['random'] = RandomPlay
Solver['rd'] = ReplicatorDynamics
