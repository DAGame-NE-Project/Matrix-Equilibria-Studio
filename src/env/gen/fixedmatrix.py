import numpy as np
import os

from util import DATA_PATH, PROJECT_PATH

def FixedMatrix(args):

    players = args.players
    actionspace = args.actionspace

    assert players == len(actionspace), "The length of actionspace should equal to #player"

    if args.isfixedpath == False:
        data_path = os.path.join(DATA_PATH, args.datapath)
    else:
        data_path = args.datapath
    with open(os.path.join(data_path, args.file_name), 'rb') as f:
        data = np.load(f)

    def generator():

        u = data.copy()
        print("u:\n", u)

        def U(a):

            assert players == len(a), "The length of jointaction should equal to #player"
            return u[tuple(a)]

        return players, actionspace, U

    return generator
