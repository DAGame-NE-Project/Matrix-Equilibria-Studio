
class SimpleRunner(object):

    def __init__(self, game, solver):
        self.game = game
        self.solver = solver
        self.empty()
        self.reset_game()

    def step(self):
        self.current_strategy, self.current_info = self.solver.solve(self.game, self.current_info)
        self.info.append(self.current_info)
        self.strategy.append(self.current_strategy)

    def episode(self, timeInterval = 1000000):
        for i in range(timeInterval):
            self.step()

    def empty(self):
        self.info = []
        self.strategy = []
        self.current_strategy = None
        self.current_info = None

    def reset_game(self):
        self.game.reset()

    def reset_solver(self):
        self.solver.reset()

    def reset(self, game = None, solver = None):
        if game is not None:
            self.game = game
        if solver is not None:
            self.solver = solver
        self.reset_game()
        self.reset_solver()
