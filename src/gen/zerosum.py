import numpy as np
import os

def ZeroSumMatrixGenerator(args):

    actionspace = args.actionspace
    players = len(actionspace)
    assert players == 2, "Zero-sum generator is only for 2-player"
    DATA_PATH = args.DATA_PATH
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    DIR_PATH = os.path.join(DATA_PATH, args.dir)
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    FILE_PATH = os.path.join(DIR_PATH, args.name)

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

    with open(FILE_PATH, 'wb') as f:
        np.save(f, u)

