from .basic_runner import BasicRunner
from .direct_runner import DirectRunner
from .iterative_runner import IterativeRunner

Runner = dict()
Runner["iterative"] = IterativeRunner
Runner["direct"] = DirectRunner
