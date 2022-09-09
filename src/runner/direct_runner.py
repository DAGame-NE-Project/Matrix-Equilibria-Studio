from .basic_runner import BasicRunner

from algo import DirectSolver
from util import epsNE_with_sample, sample_utility, show_eps, show_strategy

class DirectRunner(BasicRunner):

    def __init__(self, game, solver, args):
        super(DirectRunner, self).__init__(game, solver, args)
        self.reset_full()
        assert isinstance(self.solver, DirectSolver), "The DirectRunner should be used for direct solvers!!!"

    def run(self, record_info = None):

        eval_samples = self.args.eval_samples if hasattr(args, 'eval_samples') else 1
        utility = parse_utility(self.game, eval_samples)
        # Transpose player_id to dim 0: joint_action, player_id -> player_id, joint_action
        utility.transpose((len(utility.shape),) + tuple(range(1, len(utility.shape))))
        strategy = None
        info = None
        strategy, info = self.solver.solve(self.game, utility)

        if record_info is not None:
            # record with record_info
            record_info['record_last'] = record_overall.get('record_last', True)
            record_last_eps = []
            record_last_ws_eps = []
            ret = epsNE_with_sample(self.game, strategy, eval_samples)
            maxeps, eps = ret[0]
            record_last_eps.append((maxeps, eps))
            maxwseps, wseps = ret[1]
            record_last_ws_eps.append((maxwseps, wseps))
            output_dict = dict({})
            if record_info['record_last']:
                output_dict['last_eps'] = record_last_eps
                output_dict['last_ws_eps'] = record_last_ws_eps

            # write_to_file
            self.write_to_file(output_dict)

        if record_info['record_last']:
            show_strategy("Solution", self.game.players, strategy)
            ret = epsNE_with_sample(self.game, strategy, eval_samples)
            maxeps, eps = ret[0]
            show_eps("EPS Info", maxeps, self.game.players, eps)
            maxwseps, wseps = ret[1]
            show_eps("WSEPS Info", maxwseps, self.game.players, wseps)

        # print("last_strategy:", strategy)
        print("last_info:", info)


