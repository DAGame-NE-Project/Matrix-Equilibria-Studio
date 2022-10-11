from .basic_runner import BasicRunner

from algo import IterativeSolver
from util import epsNE_with_sample, show_eps, show_strategy

import numpy as np
from tqdm import tqdm

class IterativeRunner(BasicRunner):

    def __init__(self, game, solver, args):
        super(IterativeRunner, self).__init__(game, solver, args)
        self.reset_full()
        assert isinstance(self.solver, IterativeSolver), "The IterativeRunner should be used for iterative solvers!!!"

    def run(self):

        eval_samples = self.args.eval_samples if hasattr(self.args, 'eval_samples') else 1
        time_interval = self.args.time_interval if hasattr(self.args, 'time_interval') else 1

#        # no record
#        self.episode(time_interval)
        self.args.record_overall = getattr(self.args, 'record_overall', False)
        self.args.record_last = getattr(self.args, 'record_last', False)
        # record with record_info
        record_avg_eps = []
        record_last_eps = []
        record_avg_ws_eps = []
        record_last_ws_eps = []
        TIME_INTERVAL=1000
        record_t = []
        for time_stamp in tqdm(range(time_interval)):
            self.step()
            if time_stamp % TIME_INTERVAL == 0 or time_stamp == time_interval - 1:
                record_t.append(time_stamp)
                if self.args.record_overall:
                    strategies = list(map(lambda x: np.array(x) / (time_stamp + 1), self.current_info['overall_policy']))
                    ret = epsNE_with_sample(self.game, strategies, eval_samples)
                    maxeps, eps = ret[0]
                    record_avg_eps.append((maxeps, eps))
                    maxwseps, wseps = ret[1]
                    record_avg_ws_eps.append((maxwseps, wseps))
                if self.args.record_last:
                    strategies = self.strategy[-1]
                    ret = epsNE_with_sample(self.game, strategies, eval_samples)
                    maxeps, eps = ret[0]
                    record_last_eps.append((maxeps, eps))
                    maxwseps, wseps = ret[1]
                    record_last_ws_eps.append((maxwseps, wseps))
        output_dict = dict({})
        if self.args.record_overall or self.args.record_last:
            output_dict['time_stamp']=record_t
        if self.args.record_overall:
            output_dict['avg_eps'] = record_avg_eps
            output_dict['avg_ws_eps'] = record_avg_ws_eps
        if self.args.record_last:
            output_dict['last_eps'] = record_last_eps
            output_dict['last_ws_eps'] = record_last_ws_eps

        # write_to_file
        self._write_to_file(output_dict)

        # if need to calc average policy, please use 'overall_policy' counting the strategies in info

        if self.args.record_overall:
            strategies = list(map(lambda x: np.array(x) / time_interval, self.current_info['overall_policy']))
            show_strategy("Average Strategy", self.game.players, strategies)
            ret = epsNE_with_sample(self.game, strategies, eval_samples)
            maxeps, eps = ret[0]
            show_eps("EPS Info", maxeps, self.game.players, eps)
            maxwseps, wseps = ret[1]
            show_eps("WSEPS Info", maxwseps, self.game.players, wseps)

        # calc last_strategy (last iteration)

        if self.args.record_last:
            strategies = self.strategy[-1]
            show_strategy("Last Iteration", self.game.players, strategies)
            ret = epsNE_with_sample(self.game, strategies, eval_samples)
            maxeps, eps = ret[0]
            show_eps("EPS Info", maxeps, self.game.players, eps)
            maxwseps, wseps = ret[1]
            show_eps("WSEPS Info", maxwseps, self.game.players, wseps)

        # print("last_strategy:", self.strategy[-1])
        print("last_info:", self.info[-1])

    def step(self):
        self.current_strategy, self.current_action, self.current_info = self.solver.step(self.game, self.current_info)
        self.info.append(self.current_info)
        self.action.append(self.current_action)
        self.strategy.append(self.current_strategy)

    def episode(self, time_interval = 100000):
        for i in tqdm(range(time_interval)):
            self.step()

    def reset_history(self):
        self.info = []
        self.action = []
        self.strategy = []
        self.current_strategy = None
        self.current_action = None
        self.current_info = None

    def reset_full(self):
        self.reset_history()
        super(IterativeRunner, self).reset_full()