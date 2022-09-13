from .basic_runner import BasicRunner

from algo import DirectSolver
from util import epsNE_with_sample, sample_utility, show_eps, show_strategy

class DirectRunner(BasicRunner):

    def __init__(self, game, solver, args):
        super(DirectRunner, self).__init__(game, solver, args)
        self.reset_full()
        assert isinstance(self.solver, DirectSolver), "The DirectRunner should be used for direct solvers!!!"

    def run(self, record_info = None):

        eval_samples = self.args.eval_samples if hasattr(self.args, 'eval_samples') else 1
        utility = sample_utility(self.game, eval_samples)
        # Transpose player_id to dim 0: joint_action, player_id -> player_id, joint_action
        utility = utility.transpose((len(utility.shape) - 1, ) + tuple(range(0, len(utility.shape) - 1)))
        strategy = None
        info = None
        strategy, info = self.solver.solve(self.game, utility)

        if record_info is not None:
            # record with record_info
            record_info['record_last'] = record_info.get('record_last', True)
            record_info['record_before_adjust'] = record_info.get('record_before_adjust', True)
            record_last_eps = []
            record_last_ws_eps = []
            ret = epsNE_with_sample(self.game, strategy, eval_samples)
            maxeps, eps = ret[0]
            record_last_eps.append((maxeps, eps))
            maxwseps, wseps = ret[1]
            record_last_ws_eps.append((maxwseps, wseps))
            if 'before_adjust' in info:
                record_before_adjust_eps = []
                record_before_adjust_ws_eps = []
                ret = epsNE_with_sample(self.game, info['before_adjust'], eval_samples)
                maxeps, eps = ret[0]
                record_before_adjust_eps.append((maxeps, eps))
                maxwseps, wseps = ret[1]
                record_before_adjust_ws_eps.append((maxwseps, wseps))
            output_dict = dict({})
            if record_info['record_last']:
                output_dict['last_eps'] = record_last_eps
                output_dict['last_ws_eps'] = record_last_ws_eps
            if 'before_adjust' in info:
                output_dict['before_adjust_eps'] = record_before_adjust_eps
                output_dict['before_adjust_ws_eps'] = record_before_adjust_ws_eps

            # write_to_file
            self.write_to_file(record_info['file_name'], output_dict)

        if record_info['record_before_adjust'] and 'before_adjust' in info:
            show_strategy("Before adjust", self.game.players, info['before_adjust'])
            ret = epsNE_with_sample(self.game, info['before_adjust'], eval_samples)
            maxeps, eps = ret[0]
            show_eps("EPS Info", maxeps, self.game.players, eps)
            maxwseps, wseps = ret[1]
            show_eps("WSEPS Info", maxwseps, self.game.players, wseps)


        if record_info['record_last']:
            show_strategy("Solution", self.game.players, strategy)
            ret = epsNE_with_sample(self.game, strategy, eval_samples)
            maxeps, eps = ret[0]
            show_eps("EPS Info", maxeps, self.game.players, eps)
            maxwseps, wseps = ret[1]
            show_eps("WSEPS Info", maxwseps, self.game.players, wseps)

        # print("last_strategy:", strategy)
        print("last_info:", info)


