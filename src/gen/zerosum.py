import numpy as np

from util import DATA_PATH

def ZeroSumMatrixGenerator(args):

    actionspace = args.actionspace
    players = len(actionspace)
    assert players == 2, "Zero-sum generator is only for 2-player"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="zerosum", help="dirname of generated matrix in data, default is `zerosum`")
    parser.add_argument("--name", help="filename of generated matrix")
    parser.add_argument("--actionspace", help="actionspace of generated matrix, e.g., '[2,2]'")
    args = parser.parse_args()
    args.actionspace = eval(args.actionspace)
    ZeroSumMatrixGenerator(args)
