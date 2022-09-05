import numpy as np

# Notice that NE of a zero-sum matrix is equilvalent to that of a constant-sum one
def ZeroSumMatrixGenerator(args):

    players = args.players
    assert players == 2, "Zero-sum generator is only for 2-player"
    actionspace = args.actionspace

    assert players == len(actionspace), "The length of actionspace should equal to #player"

    def generator():

        u = []
        ui = np.random.random(actionspace)
        if ui.min() == ui.max():
            ui = np.ones_like(ui)
        else:
            ui = ui - ui.min()
            ui = ui / ui.max()
        u.append(np.expand_dims(ui, players))
        u.append(np.expand_dims(1 - ui, players))
        u = np.concatenate(u, players)
        print("u:\n", u)

        def U(a):

            assert players == len(a), "The length of jointaction should equal to #player"
            return u[tuple(a)]

        return players, actionspace, U

    return generator
