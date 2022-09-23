import numpy as np

from util import DATA_PATH

def RandomMatrixGenerator(args):

    actionspace = args.actionspace
    players = len(actionspace)
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    DIR_PATH = os.path.join(DATA_PATH, args.dir)
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)
    FILE_PATH = os.path.join(DIR_PATH, args.name)

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

    with open(FILE_PATH, 'wb') as f:
        np.save(f, u)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="random", help="dirname of generated matrix in data, default is `random`")
    parser.add_argument("--name", help="filename of generated matrix")
    parser.add_argument("--actionspace", help="actionspace of generated matrix, e.g., '[2,2]'")
    args = parser.parse_args()
    args.actionspace = eval(args.actionspace)
    RandomMatrixGenerator(args)
