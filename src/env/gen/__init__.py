from .randommatrixgenerator import RandomMatrixGenerator
from .zerosummatrixgenerator import ZeroSumMatrixGenerator
from .fixedmatrix import FixedMatrix

GameGenerator = dict()
GameGenerator['random'] = RandomMatrixGenerator
GameGenerator['zerosum'] = ZeroSumMatrixGenerator
GameGenerator['fixed'] = FixedMatrix
