from .Hedge import Hedge
from .FictitiousPlay import Player as FictitiousPlay
from .MW import MW as MWU
from .RandomPlay import RandomPlay
Solver = dict()

Solver['random'] = RandomPlay
Solver['fp'] = FictitiousPlay
Solver['mwu'] = MWU
Solver['hedge'] = Hedge
