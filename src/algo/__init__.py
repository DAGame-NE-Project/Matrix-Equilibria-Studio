from .Hedge import Hedge
from .FictitiousPlay import Player as FictitiousPlay
from .MW import MW as MWU
from .RandomPlay import RandomPlay
from .RM import Player as RM
Solver = dict()

Solver['random'] = RandomPlay
Solver['fp'] = FictitiousPlay
Solver['mwu'] = MWU
Solver['hedge'] = Hedge
Solver['rm'] = RM
