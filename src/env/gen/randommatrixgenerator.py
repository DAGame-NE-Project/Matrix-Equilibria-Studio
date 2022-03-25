import numpy as np

def RandomMatrixGenerator(players, actionspace):

    assert players == len(actionspace), "The length of actionspace should equal to #player"

    def generator():

        u = []
        for i in range(players):
            ui = np.random.random(actionspace)
            if ui.min() == ui.max():
                ui = np.ones_like(ui)
            else:
                ui = ui - ui.min()
                ui = ui / ui.max()
            u.append(np.expand_dims(ui, players))
        u = np.concatenate(u, players)

        def U(a):

            assert players == len(a), "The length of jointaction should equal to #player"
            return u(tuple(a))

        return players, actionspace, U

    return generator
