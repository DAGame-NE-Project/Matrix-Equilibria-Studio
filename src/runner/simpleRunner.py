from tqdm import tqdm
class SimpleRunner(object):

    def __init__(self, game, solver, args):
        self.game = game
        self.solver = solver
        self.args = args
        self.reset_full()

    def step(self):
        self.current_strategy, self.current_action, self.current_info = self.solver.solve(self.game, self.current_info)
        self.info.append(self.current_info)
        self.action.append(self.current_action)
        self.strategy.append(self.current_strategy)

    def episode(self, timeInterval = 100000):
        for i in tqdm(range(timeInterval)):
            self.step()

    def reset_history(self):
        self.info = []
        self.action = []
        self.strategy = []
        self.current_strategy = None
        self.current_action = None
        self.current_info = None

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
        self.reset_history()
        self.reset_runner()
