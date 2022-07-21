from .RandomPlay import RandomPlay
from .FictitiousPlay import Player as FictitiousPlay
from .MW import MW as MWU
from .ReplicatorDynamics import ReplicatorDynamics
Solver = dict()

Solver['random'] = RandomPlay
Solver['fp'] = FictitiousPlay
Solver['mwu'] = MWU
Solver['rd'] = ReplicatorDynamics
