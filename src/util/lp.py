import numpy as np
import scipy as sp
import scipy.optimize

def solve_lp(A_le = None, b_le = None, A_eq = None, b_eq = None, A_ge = None, b_ge = None, c = None, object_func = 'min', bounds = (0, None), method = 'highs-ds'):
    # Hint: if using scipy 1.9.0, please replace "revised simplex" by "highs-ds"
    assert (c is not None)
    assert (object_func in ['min', 'max'])
    A = []
    b = []
    if A_le is not None:
        assert (b_le is not None)
        assert (len(A_le) == len(b_le))
        A.append(A_le)
        b.append(b_le)
    if A_eq is not None:
        assert (b_eq is not None)
        assert len(A_eq) == len(b_eq)
    if A_ge is not None:
        assert (b_ge is not None)
        assert (len(A_ge) == len(b_ge))
        A.append(-A_ge)
        b.append(-b_ge)
    A = np.concatenate(A, axis = 0)
    b = np.concatenate(b, axis = 0)
    if object_func != 'min':
        c = -c
    sol = sp.optimize.linprog(c = c, A_ub = A, b_ub = b, A_eq = A_eq, b_eq = b_eq, method = method, bounds = bounds)
    if object_func != 'min':
        sol.fun = -sol.fun
    return sol

def check_lp(x, real_sol = None, A_le = None, b_le = None, A_eq = None, b_eq = None, A_ge = None, b_ge = None, c = None, object_func = 'min', bounds = (0, None), EPS = 1e-10):
    xlen = len(x)
    assert(c is not None)
    if object_func != 'min':
        c = -c
    if bounds is not list:
        bounds = [bounds for i in xlen]
    assert(len(bounds) == x)
    for i in range(xlen):
        if not bounds[i][0] - EPS <= x[i] <= bounds[i][1] + EPS:
            return False, "bounds error"
    if A_le is not None:
        if not np.all(A_le @ x <= b_le + EPS):
            return False, "le_constrains error"
    if A_eq is not None:
        if not np.all(np.abs(A_eq @ x - b_eq) <= EPS):
            return False, "eq_constrains error"
    if A_ge is not None:
        if not np.all(A_ge @ x >= b_ge - EPS):
            return False, "ge_constrains error"
    if real_sol is None:
        real_sol = solve_lp(A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq, A_ge = A_ge, b_ge = b_ge, c = c, object_func = object_func, bounds = bounds)
    if real_sol.status in (3, 4):
        return False, "lp_status_code {}".format(real_sol.status)
    if c @ x > real_sol.fun + EPS:
        if object_func == 'max':
            c = -c
            real_sol.fun = -real_sol.fun
        return False, "value error real {} provided {} with {} function".format(real_sol.fun, c @ x, object_func)
    return True, "successfully"

def solve_zero_sum(R, check_result = False, EPS = 1e-10):
    # solve zero sum game (R, -R), R is an (n, m) real-number matrix
    # return a tuple (row/max player: x, col/min player: y, game value: v)
    R = np.array(R)
    n = len(R)
    m = len(R[0])
    base = np.zeros((1, n + m + 2))
    # x y v_+ v_-
    # (xR)^T >= v_+ - v_- => R^Tx^T - v_+ + v_- >= 0
    A_ge = np.concatenate([R.T, np.zeros((m, m)), np.ones((m, 1)) @ [[-1, 1]]], axis = 1)
    b_ge = np.zeros(m)
    # Ry <= v_+ - v_- => Ry - v_+ + v_- <= 0
    A_le = np.concatenate([np.zeros((n,n)), R, np.ones((n, 1)) @ [[-1, 1]]], axis = 1)
    b_le = np.zeros(n)
    # sum(x) = 1, sum(y) = 1
    A_eq = []
    A_eq.append(np.concatenate([np.ones((1, n)), np.zeros((1, m)), np.zeros((1, 2))], axis = 1))
    A_eq.append(np.concatenate([np.zeros((1, n)), np.ones((1, m)), np.zeros((1, 2))], axis = 1))
    A_eq = np.concatenate(A_eq, axis = 0)
    b_eq = np.ones(2)
    # max v_+ - v_-
    c = np.concatenate([np.zeros(n + m), np.array([1, -1])], axis = 0)
    object_func = 'max'
    sol = solve_lp(A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq, A_ge=A_ge,
                   b_ge=b_ge, c=c, object_func=object_func, method='highs-ipm')
    if check_result:
        check_res = check_lp(sol.x, sol, A_le = A_le, b_le = b_le, A_eq = A_eq, b_eq = b_eq, A_ge = A_ge, b_ge = b_ge, c = c, object_func = object_func, EPS = EPS)
        if not check_res[0]:
            print("R:\n")
            print(R)
            print("sol:", sol.x)
            print("c:", sol.fun)
            print(check_res[1])
            raise Exception(check_res[1])
    ret_x = sol.x[:n].copy()
    ret_y = sol.x[n:n+m].copy()
    ret_v = sol.fun
    return (ret_x, ret_y, ret_v)

if __name__ == "__main__":
    R = np.array([[0,1,-1], [-1,0,1], [1,-1,0]]) ## RPS game
    x,y,v = solve_zero_sum(R)
    print("RPS:")
    print("row player:", x)
    print("col player:", y)
    print("game value:", v)
    R = np.array([[1,-1],[-1,1]]) ## Matching Pennis
    x,y,v = solve_zero_sum(R)
    print("Matching Pennis:")
    print("row player:", x)
    print("col player:", y)
    print("game value:", v)
    R = np.array([[0, 1], [-1, 0]])
    x,y,v = solve_zero_sum(R) ## (0,0) dominance
    print("Dominance Solvable:")
    print("row player:", x)
    print("col player:", y)
    print("game value:", v)

