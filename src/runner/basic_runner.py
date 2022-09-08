import abc
import json
import os

class BasicRunner(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, game, solver, args):
        self.game = game
        self.solver = solver
        self.args = args

    @abc.abstractmethod
    def run(self, record_info = None):
        pass

    def write_to_file(self, output_dict):

        RESULT_PATH = self.args.resultpath
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)
        SUBDIR_PATH = os.path.join(RESULT_PATH, args.solver)
        if not os.path.exists(SUBDIR_PATH):
            os.mkdir(SUBDIR_PATH)
        with open(os.path.join(SUBDIR_PATH, record_info['file_name'] + ".json"), 'w') as fp:
            json.dump(output_dict, fp, separators=(",\n",":\n"))

    def reset_game(self):
        self.game.reset()

    def reset_solver(self):
        self.solver.reset()

    def reset_runner(self, game = None, solver = None):
        if game is not None:
            self.game = game
        if solver is not None:
            self.solver = solver
        self.reset_game()
        self.reset_solver()

    def reset_full(self):
        self.reset_runner()
