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
    def run(self):
        pass

    def write_to_file(self, file_name, output_dict):

        with open(file_name, 'w') as fp:
            json.dump(output_dict, fp, separators=(",\n",":\n"))

    def _write_to_file(self, output_dict):

        self.write_to_file(os.path.join(self.args.resultpath, self.args.file_name + ".json"), output_dict)

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
