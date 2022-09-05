from .randommatrixgenerator import RandomMatrixGenerator
from .zerosummatrixgenerator import ZeroSumMatrixGenerator

GameGenerator = dict()
GameGenerator['random'] = RandomMatrixGenerator
GameGenerator['zerosum'] = ZeroSumMatrixGenerator
