import numpy as np
from itertools import combinations_with_replacement

#'epsNE', 'epsWSNE'
compares = {}


def multisupp2strategy(multisupp, total_action_num):
    strategy = np.zeros(total_action_num)
    for i in multisupp:
        strategy[i] += 1
    strategy = strategy/np.sum(strategy)
    return strategy


def exhaustive_search(R, C, k, comp):
    # exhaustive_search on k-uniform strategies
    action_num_x, action_num_y = R.shape
    ret_x, ret_y = None, None
    for multisupp_x in combinations_with_replacement(range(action_num_x), k):
        for multisupp_y in combinations_with_replacement(range(action_num_y), k):
            x = multisupp2strategy(multisupp_x, action_num_x)
            y = multisupp2strategy(multisupp_y, action_num_y)
            if ret_x is None:
                ret_x = x
                ret_y = y
            elif compares[comp](x, y, ret_x, ret_y, R, C):
                ret_x = x
                ret_y = y
    return (ret_x, ret_y)


######################## Compare methods ############################

def eps_comp(x1, y1, x2, y2, R, C):
    # compare eps-NE of (x1,y1) and (x2,y2)
    def regret(x, y):
        return max(np.max(R @ y) - x @ R @ y, np.max(x @ C) - x @ C @ y)
    return regret(x1, y1) < regret(x2, y2)


def epsWS_comp(x1, y1, x2, y2, R, C, ):
    # compare eps-WSNE of (x1,y1) and (x2,y2)
    NON_ZERO = 1e-10

    def regret(x, y):
        regret_x = np.max(
            R @ y) - np.min(np.where(x > NON_ZERO, R @ y, np.ones_like(x)))
        regret_y = np.max(x @ C) - np.min(np.where(y >
                                                   NON_ZERO, x @ C, np.ones_like(y)))
        return max(regret_x, regret_y)
    return regret(x1, y1) < regret(x2, y2)


compares['epsNE'] = eps_comp
compares['epsWSNE'] = epsWS_comp

if __name__ == '__main__':
    # test all functions
    # test multisupp2strategy
    x = np.array([1, 1, 0, 3, 2])
    print(multisupp2strategy(x, 6))

    # test eps_comp
    # extended matching pennies
    R = np.array([[1, 0, 0], [0, 1, 0]])
    C = np.array([[0, 1, 0], [1, 0, 0]])
    x1 = np.array([0, 1])
    y1 = np.array([0, 0, 1])
    x2 = np.array([0.4, 0.6])
    y2 = np.array([0.5, 0.5, 0])
    print(eps_comp(x1, y1, x2, y2, R, C))

    # test epsWS_comp
    print(epsWS_comp(x1, y1, x2, y2, R, C))

    # test exhaustive_search
    print(exhaustive_search(R, C, 4, 'epsNE'))
    print(exhaustive_search(R, C, 4, 'epsWSNE'))
    R = C
    print(exhaustive_search(R, C, 1, 'epsNE'))
    print(exhaustive_search(R, C, 1, 'epsWSNE'))
    # paper-scissors-rock
    R = np.array([[0.5,1,0],[0,0.5,1],[1,0,0.5]])
    C = 1 - R
    print(exhaustive_search(R, C, 1, 'epsNE'))
    print(exhaustive_search(R, C, 1, 'epsWSNE'))
    print(exhaustive_search(R, C, 2, 'epsNE'))
    print(exhaustive_search(R, C, 2, 'epsWSNE'))
    print(exhaustive_search(R, C, 3, 'epsNE'))
    print(exhaustive_search(R, C, 3, 'epsWSNE'))