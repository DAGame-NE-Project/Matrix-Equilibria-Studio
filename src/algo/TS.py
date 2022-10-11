from util.lp import check_lp, solve_lp
import numpy as np
import scipy as sp
from .direct_solver import DirectSolver


# 'constant', 'adapted'
line_search_methods = {}

# 'TS', 'linearize', 'DFM'
adjust_methods = {}

TOTAL_FLOAT_VALUE_EPS = 1e-12


class Player(DirectSolver):
    def __init__(self, args):
        super(Player, self).__init__(args)
        self.line_search_method = args.line_search_method if hasattr(
            args, 'line_search_method') else 'adapted'
        self.adjust_method = args.adjust_method if hasattr(
            args, 'adjust_method') else 'DFM'
        self.delta = args.delta if hasattr(args, 'delta') else 0.1
        self.n_iter = args.n_iter if hasattr(args, 'n_iter') else None
        self.init_x = args.init_x if hasattr(args, 'init_x') else None
        self.init_y = args.init_y if hasattr(args, 'init_y') else None

    def solve(self, game, utility):
        actions = game.getActionSpace()
        players = game.players
        assert players == 2, "TS only works for 2-player games!"
        ret = []

        for player_id in range(players):
            ret.append(np.zeros(actions[player_id]))

        R, C = utility[0], utility[1]
        res = solve(R=R, C=C, adjust_method=self.adjust_method, line_search_method=self.line_search_method,
                    delta=self.delta, n_iter=self.n_iter, init_x=self.init_x, init_y=self.init_y)
        x_s, y_s, ret[0], ret[1] = res
        info = {
            'solver': "TS",
            'adjust_method': self.adjust_method,
            'line_search_method': self.line_search_method,
            'delta': self.delta,
            'overall_policy': [ret[player_id].copy() for player_id in range(players)],
            'strategy_before_adjust': [x_s.copy(), y_s.copy()],
        }
        return ret, info

    def reset(self):
        pass


# run TS algorithm
def solve(R, C, adjust_method, line_search_method, delta=0.1, n_iter=None, init_x=None, init_y=None, record_points=False):
    rec_x = []
    rec_y = []
    rec_a = []
    rec_fR = []
    rec_fC = []
    rec_f = []
    rec_w = []
    rec_z = []
    eps = delta / (1 + delta)
    x = np.array([np.random.random() for i in range(R.shape[0])])
    x = x / np.sum(x)
    if init_x is not None:
        x = init_x
    y = np.array([np.random.random() for i in range(R.shape[1])])
    y = y / np.sum(y)
    if init_y is not None:
        y = init_y
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    extra = extra_info(R, C, x, y, delta)
    # run ts
    step = 0
    if record_points:
        rec_x.append(x.flatten().tolist())
        rec_y.append(y.flatten().tolist())
        rec_w.append(extra.w.flatten().tolist())
        rec_z.append(extra.z.flatten().tolist())
        rec_a.append("init")
        rec_fR.append(float(extra.fR))
        rec_fC.append(float(extra.fC))
        rec_f.append(float(extra.f))
    while True:
        step += 1
        if step % 1000 == 0:
            print(step)
            print(calculate_f_value(R, C, x, y))
        # equal regret adjust
        if float_equal(extra.fR, extra.fC):
            pass
        elif extra.fR > extra.fC:
            x = solve_eq(extra.CT, extra.RT, y, extra.m, extra.brx['value'])
            extra.update(x, y)
            if record_points:
                rec_x.append(x.flatten().tolist())
                rec_y.append(y.flatten().tolist())
                rec_w.append(extra.w.flatten().tolist())
                rec_z.append(extra.z.flatten().tolist())
                rec_a.append("fR>fC")
                rec_fR.append(float(extra.fR))
                rec_fC.append(float(extra.fC))
                rec_f.append(float(extra.f))
        else:
            y = solve_eq(extra.R, extra.C, x, extra.n, extra.bry['value'])
            extra.update(x, y)
            if record_points:
                rec_x.append(x.flatten().tolist())
                rec_y.append(y.flatten().tolist())
                rec_w.append(extra.w.flatten().tolist())
                rec_z.append(extra.z.flatten().tolist())
                rec_a.append("fC>fR")
                rec_fR.append(float(extra.fR))
                rec_fC.append(float(extra.fC))
                rec_f.append(float(extra.f))
        sols = max_gradient_direction(x, y, extra)
        V = sols['primary'].fun
        tmp_diff = (V - extra.f) / delta
        if V - extra.f > -delta:
            rec_a.append("gradient stop %.5f delta" % tmp_diff)
            break
        eps_adjust_args = {
            "eps": eps,
            "sols": sols,
            "V": sols['primary'].fun,
        }
        x, y = line_search_methods[line_search_method](
            x, y, extra, eps_adjust_args)
        extra.update(x, y)
        if record_points:
            rec_x.append(x.flatten().tolist())
            rec_y.append(y.flatten().tolist())
            rec_w.append(extra.w.flatten().tolist())
            rec_z.append(extra.z.flatten().tolist())
            rec_a.append("gradient %.5f delta" % tmp_diff)
            rec_fR.append(float(extra.fR))
            rec_fC.append(float(extra.fC))
            rec_f.append(float(extra.f))
        if n_iter is not None and step >= n_iter:
            break
    # last adjust
    x_s, y_s = x.flatten(), y.flatten()
    res = adjust_methods[adjust_method](R, C, x, y, extra)  # a dict
    if record_points:
        res['record'] = {
            "x": rec_x,
            "y": rec_y,
            "w": rec_w,
            "z": rec_z,
            "method": rec_a,
            "fR": rec_fR,
            "fC": rec_fC,
            "f": rec_f,
        }
    return x_s, y_s, res['x'].flatten(), res['y'].flatten()


########################### line search #############################

def constant_line_search(x, y, extra, args):
    eps = args['eps']
    x = x + (extra.xp - x) * eps
    y = y + (extra.yp - y) * eps
    return x, y


def adapted_line_search(x, y, extra, args):
    eps = 1.
    Ryp = extra.R @ extra.yp
    mvRyp = float(np.max([Ryp[i] for i in extra.brx['index']]))
    CTxp = extra.CT @ extra.xp
    mvCTxp = float(np.max([CTxp[i] for i in extra.bry['index']]))
    for i in range(extra.m):
        if i not in extra.brx['index']:
            numer = float(extra.brx['value'] - extra.Ry[i])
            domin = float(numer + Ryp[i] - mvRyp)
            if numer < domin and domin > 0.:
                eps = min(eps, numer / domin)
    for i in range(extra.n):
        if i not in extra.bry['index']:
            numer = float(extra.bry['value'] - extra.CTx[i])
            domin = float(numer + CTxp[i] - mvCTxp)
            if numer < domin and domin > 0:
                eps = min(eps, numer / domin)
    H = np.min([
        (extra.xp - x).transpose() @ extra.R @ (extra.yp - y),
        (extra.xp - x).transpose() @ extra.C @ (extra.yp - y)])
    if H < 0.:
        numer = abs(args['V'] - extra.f)
        domin = -2. * H
        if numer < domin and domin > 0.:
            eps = min(eps, numer / domin)
    x = x + (extra.xp - x) * eps
    y = y + (extra.yp - y) * eps
    return x, y


line_search_methods['constant'] = constant_line_search
line_search_methods['adapted'] = adapted_line_search


######################## adjustment methods ##########################

def ts_adjust(R, C, x, y, extra):

    def calculate_lamb_mu(x, y, extra):
        tempR = get_matrix_by_index(
            extra.RT @ (extra.w - x), row_index=extra.bry['index'])
        lamb = get_best_response(tempR, best_func=np.min)['value']
        tempC = get_matrix_by_index(
            extra.C @ (extra.z - y), row_index=extra.brx['index'])
        mu = get_best_response(tempC, best_func=np.min)['value']
        return (lamb, mu)

    def calculate_tilde_vec(x, y, w, z, lamb, mu):
        if lamb > mu:
            return (x + (w - x) / (1 + lamb - mu), z)
        else:
            return (w, y + (z - y) / (1 - mu + lamb))

    lamb, mu = calculate_lamb_mu(x, y, extra)
#    print("lamb: ", lamb, ", mu: ", mu)
#    print("w: ", extra.w, "z: ", extra.z)
    xt, yt = calculate_tilde_vec(x, y, extra.w, extra.z, lamb, mu)
    x_sol, y_sol = x, y
    f_sol = extra.f
    f_tmp = calculate_f_value(R, C, xt, yt)['f']
    if f_tmp < f_sol:
        x_sol, y_sol, f_sol = xt, yt, f_tmp
        x, y = xt, yt
    return {'x': x_sol, 'y': y_sol, 'approximation': f_sol}


def linear_adjust(R, C, x, y, extra):

    def calculate_tilde_vec(x, y, w, z, p, q, fwz):
        if fwz['fC'] >= fwz['fR']:
            return (p * w + (1 - p) * x, z)
        else:
            return (w, q * z + (1 - q) * y)

    fxz = calculate_f_value(R, C, x, extra.z)
    fwy = calculate_f_value(R, C, extra.w, y)
    fwz = calculate_f_value(R, C, extra.w, extra.z)
    if fxz['fR'] + fwz['fC'] - fwz['fR']:
        p = fxz['fR'] / (fxz['fR'] + fwz['fC'] - fwz['fR'])
    else:
        p = 0
    if fwy['fC'] + fwz['fR'] - fwz['fC']:
        q = fwy['fC'] / (fwy['fC'] + fwz['fR'] - fwz['fC'])
    else:
        q = 0
    xt, yt = calculate_tilde_vec(x, y, extra.w, extra.z, p, q, fwz)
    x_sol, y_sol = x, y
    f_sol = extra.f
    f_tmp = calculate_f_value(R, C, xt, yt)['f']
    if f_tmp < f_sol:
        x_sol, y_sol, f_sol = xt, yt, f_tmp
        x, y = xt, yt
    return {'x': x_sol, 'y': y_sol, 'approximation': f_sol}


def dfm_adjust(R, C, x, y, extra):
    def calculate_lamb_mu(x, y, extra):
        lamb = (extra.w - x).transpose() @ extra.R @ extra.z
        mu = extra.w.transpose() @ extra.C @ (extra.z - y)
        return (lamb[0, 0], mu[0, 0])
    lamb, mu = calculate_lamb_mu(x, y, extra)
    x_sol, y_sol = x, y
    f_sol = extra.f
    # use (x, y)
    if min(lamb, mu) <= 1/2 or max(lamb, mu) <= 2/3:
        pass
    # use (w,z)
    elif min(lamb, mu) >= 2/3:
        x_sol, y_sol = extra.w, extra.z
        f_sol = calculate_f_value(R, C, x_sol, y_sol)
    # adjust
    elif 1/2 < lamb <= 2/3 < mu:
        haty = (y+extra.z)/2
        hatw = np.zeros_like(x).reshape(-1, 1)
        supphatw = get_best_response(R @ haty, best_func=np.max)['index'][0]
        hatw[supphatw, 0] = 1.0
        t_r = (hatw.transpose() @ R @ haty -
               extra.w.transpose() @ R @ haty)[0, 0]
        v_r = (extra.w.transpose() @ R @ y - hatw.transpose() @ R @ y)[0, 0]
        hatmu = (hatw.transpose() @ C @ extra.z -
                 hatw.transpose() @ C @ y)[0, 0]
        xt, yt = x, y
        if v_r+t_r >= (lamb-mu)/2 and hatmu >= mu-v_r-t_r:
            p = 1-(mu-lamb)/(2*(v_r+t_r))
            xt = p * extra.w+(1-p)*hatw
            yt = extra.z
        else:
            q = (1-mu/2-t_r)/(1+mu/2-lamb-t_r)
            xt = extra.w
            yt = (1-q)*haty + q*extra.z
        f_t = calculate_f_value(R, C, xt, yt)['f']
        if f_t < f_sol:
            x_sol, y_sol = xt, yt
            f_sol = f_t
    # symmetrically adjusted
    else:
        hatx = (x+extra.w)/2
        hatz = np.zeros_like(y)
        supphatx = get_best_response(
            extra.CT @ hatx, best_func=np.max)['index'][0]
        hatx[supphatx] = 1.0
        t_c = (hatx.transpose() @ R @ hatz -
               hatx.transpose() @ R @ extra.z)[0, 0]
        v_c = (x.transpose() @ C @ extra.z-x.transpose() @ C @ hatz)[0, 0]
        hatlamb = (extra.w.transpose() @ R @ hatz -
                   x.transpose() @ R @ hatz)[0, 0]
        xt, yt = x, y
        if v_c+t_c >= (lamb-mu)/2 and hatlamb >= lamb-v_c-t_c:
            xt = extra.w
            yt = p * extra.z + (1-p)*hatz
        else:
            q = (1-lamb/2-t_c)/(1+lamb/2-mu-t_c)
            xt = (1-q)*hatx+q*extra.w
            yt = extra.z
        f_t = calculate_f_value(R, C, xt, yt)['f']
        if f_t < f_sol:
            x_sol, y_sol = xt, yt
            f_sol = f_t
    return {'x': x_sol, 'y': y_sol, 'approximation': f_sol}


adjust_methods['TS'] = ts_adjust
adjust_methods['linearize'] = linear_adjust
adjust_methods['DFM'] = dfm_adjust


##################### functions for descent procedure #######################

def solve_eq(R, C, x, n, b_base):  # calculate y for equal regret
    xTC = x.transpose() @ C
    A_le = []
    tmp_x = x.transpose().copy()
    for i in range(x.shape[0]):
        tmp_x[0][i] -= 1
        A_le.append(-tmp_x @ R + xTC)
        tmp_x[0][i] += 1
    A_le = np.concatenate(A_le, axis=0)
    b_le = np.ones(x.shape[0]) * b_base
    A_eq = np.ones((1, n))
    b_eq = np.ones(1)
    c = -xTC.flatten()
    res = solve_lp(A_le=A_le, b_le=b_le, A_eq=A_eq, b_eq=b_eq, c=c)
    return res.x.reshape((n, 1))


def build_G_matrix(x, y, extra):
    e_m = extra.e_m
    e_n = extra.e_n
    G = np.concatenate([
        np.concatenate([
            extra.R - e_m @ extra.xTR,
            - e_m @ y.transpose() @ extra.RT + e_m @ e_m.transpose() * (extra.xTRy)
        ], axis=1),
        np.concatenate([
            - e_n @ x.transpose() @ extra.C + e_n @ e_n.transpose() * (extra.xTCy),
            extra.CT - e_n @ extra.yTCT
        ], axis=1)
    ], axis=0)
    return G


def max_gradient_direction(x, y, extra):
    G = build_G_matrix(x, y, extra)
    # solve_origin_lp
    index_wz = np.concatenate(
        [extra.brx['index'], extra.bry['index'] + extra.m])
    G_tmp = get_matrix_by_index(G, row_index=index_wz)
    A_ge = np.concatenate([
        -G_tmp,
        np.ones((G_tmp.shape[0], 1)),
        np.ones((G_tmp.shape[0], 1)) * -1
    ], axis=1)
    b_ge = np.zeros(A_ge.shape[0])
    A_eq = np.concatenate([
        np.ones(extra.n),
        np.zeros(extra.m),
        np.zeros(2),
        np.zeros(extra.n),
        np.ones(extra.m),
        np.zeros(2)
    ]).reshape((2, extra.n + extra.m + 2))
    b_eq = np.ones(2)
    c = np.concatenate([np.zeros(extra.n + extra.m), [1, -1]])
    pri_sol = solve_lp(A_ge=A_ge, b_ge=b_ge, A_eq=A_eq, b_eq=b_eq, c=c)
    # extract xp, yp
    xp = pri_sol.x[extra.n: extra.m + extra.n].copy()
    yp = pri_sol.x[: extra.n].copy()
    extra.update_xpyp(xp, yp)
    # solve_dual_lp
    idx_w = extra.brx['index']
    len_w = idx_w.shape[0]
    idx_z = extra.bry['index']
    len_z = idx_z.shape[0]
    G_tmpT = G_tmp.transpose()
    coef_delta_y = np.concatenate([
        np.ones((extra.n, 1)),
        np.zeros((extra.m, 1))
    ])
    coef_delta_x = np.ones((extra.m + extra.n, 1)) - coef_delta_y
    A_le = np.concatenate([
        G_tmpT,
        coef_delta_x,
        -coef_delta_x,
        coef_delta_y,
        -coef_delta_y
    ], axis=1)
    b_le = np.zeros(extra.m + extra.n)
    A_eq = np.concatenate([
        np.ones((1, len_w + len_z)),
        np.zeros((1, 4))
    ], axis=1)
    b_eq = np.ones(1)
    c = np.concatenate([
        np.zeros(len_w + len_z),
        [1, -1, 1, -1]
    ])
    dual_sol = solve_lp(A_le=A_le, b_le=b_le, A_eq=A_eq,
                        b_eq=b_eq, c=c, object_func='max')
    # extract w, z
    w = np.zeros(extra.m)
    for i in range(len_w):
        w[idx_w[i]] = dual_sol.x[i]
    z = np.zeros(extra.n)
    for i in range(len_z):
        z[idx_z[i]] = dual_sol.x[len_w + i]
    extra.update_wz(w, z)
    return {'primary': pri_sol, 'dual': dual_sol}


########################### auxiliary functions #############################

def calculate_f_value(R, C, x, y):
    Ry = R @ y
    brx = get_best_response(Ry, best_func=np.max)
    fR = float(brx['value'] - x.transpose() @ Ry)
    CTx = C.transpose() @ x
    bry = get_best_response(CTx, best_func=np.max)
    fC = float(bry['value'] - y.transpose() @ CTx)
    return {'fR': fR, 'fC': fC, 'f': np.max([fR, fC])}


def normalize_matrix(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x


def unify_vec(x):
    if np.sum(x) <= TOTAL_FLOAT_VALUE_EPS:
        x = np.ones(x.shape)
    return x / np.sum(x)


# best response with index array, best_func need to be assigned as np.max or np.min
def get_best_response(Ax, best_func, delta=TOTAL_FLOAT_VALUE_EPS):
    b = Ax.flatten()
    best_v = best_func(b)
    return {'value': best_v, 'index': np.argwhere(b >= best_v - delta).flatten()}


class extra_info:
    def __init__(self, R, C, x, y, delta):  # x, y are column vectors
        self.m = x.shape[0]
        self.n = y.shape[0]
        self.w = np.zeros_like(x)
        self.z = np.zeros_like(y)
        self.e_m = np.ones(x.shape)
        self.e_n = np.ones(y.shape)
        self.R = R.copy()
        self.RT = R.transpose().copy()
        self.C = C.copy()
        self.CT = C.transpose().copy()
        self.delta = delta
        self.update(x, y)

    def update(self, x, y):
        self.xTR = x.transpose() @ self.R
        self.yTCT = y.transpose() @ self.CT
        self.Ry = self.R @ y
        self.CTx = self.CT @ x
        self.xTRy = self.xTR @ y
        self.xTCy = self.yTCT @ x
        self.brx = get_best_response(
            self.Ry, best_func=np.max, delta=self.delta)
#        print("===Best_Response_x===")
#        print(self.Ry)
#        print(self.brx['index'])
#        print("========END========")
        self.bry = get_best_response(
            self.CTx, best_func=np.max, delta=self.delta)
#        print("===Best_Response_y===")
#        print(self.CTx)
#        print(self.bry['index'])
#        print("========END========")
        self.fR = self.brx['value'] - self.xTRy
        self.fC = self.bry['value'] - self.xTCy
        self.f = np.max([self.fR, self.fC])

    def update_xpyp(self, xp, yp):
        self.xp = xp.reshape((self.m, 1))
        self.yp = yp.reshape((self.n, 1))

    def update_wz(self, w, z):
        #        print("ori w: ", w, "ori z: ", z)
        self.w = unify_vec(w.reshape((self.m, 1)))
        self.z = unify_vec(z.reshape((self.n, 1)))


# select row_index and col_index, None means no selection
def get_matrix_by_index(A, row_index=None, col_index=None):
    if row_index is not None:
        A = np.concatenate(
            list(map(lambda x: A[x].reshape(1, A.shape[1]), list(row_index))))
    if col_index is not None:
        A = np.concatenate(list(map(lambda x: A.T[x].reshape(
            1, A.shape[0]), list(col_index)))).transpose()
    return A


def get_support(x):
    return np.argwhere(x != 0.0)


def float_equal(x, y, eps=TOTAL_FLOAT_VALUE_EPS):
    return x <= y + eps and y <= x + eps


# check solution of TS algorithm
def check_solution(R, C, x, y, w, z, rho, EPS=1e-10):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    w = w.reshape(-1, 1)
    extra = extra_info(R, C, x, y, EPS)
    # check f
    if not float_equal(extra.fR, extra.fC, EPS):
        return False, "fR is not equal to fC"
    sols = max_gradient_direction(x, y, extra)
    G = build_G_matrix(x, y, extra)
    # check origin_lp
    index_wz = np.concatenate(
        [extra.brx['index'], extra.bry['index'] + extra.m])
    G_tmp = get_matrix_by_index(G, row_index=index_wz)
    A_ge = np.concatenate([
        -G_tmp,
        np.ones((G_tmp.shape[0], 1)),
        np.ones((G_tmp.shape[0], 1)) * -1
    ], axis=1)
    b_ge = np.zeros(A_ge.shape[0])
    A_eq = np.concatenate([
        np.ones(extra.n),
        np.zeros(extra.m),
        np.zeros(2),
        np.zeros(extra.n),
        np.ones(extra.m),
        np.zeros(2)
    ]).reshape((2, extra.n + extra.m + 2))
    b_eq = np.ones(2)
    c = np.concatenate([np.zeros(extra.n + extra.m), [1, -1]])
    pri_sol = solve_lp(A_ge=A_ge, b_ge=b_ge, A_eq=A_eq, b_eq=b_eq, c=c)
    vec_xy = np.concatenate([y, x], axis=0).flatten()
    v_xy = np.max(G_tmp @ vec_xy)
    if pri_sol.status in (3, 4):
        return False, "lp_status_code {}".format(pri_sol.status)
    # print("xy diff\n",pri_sol.x[:extra.n + extra.m] - vec_xy)
    if v_xy > pri_sol.fun + EPS:
        return False, "value error real {} v_xy {} with {} function\n solution {}\n xy is {} {}\nf_xy is {}".format(pri_sol.fun, v_xy, "min", pri_sol.x, y.flatten(), x.flatten(), calculate_f_value(R, C, x, y)['f'])
    # solve_dual_lp
    idx_w = extra.brx['index']
    len_w = idx_w.shape[0]
    idx_z = extra.bry['index']
    len_z = idx_z.shape[0]
    G_tmpT = G_tmp.transpose()
    coef_delta_y = np.concatenate([
        np.ones((extra.n, 1)),
        np.zeros((extra.m, 1))
    ])
    coef_delta_x = np.ones((extra.m + extra.n, 1)) - coef_delta_y
    A_le = np.concatenate([
        -G_tmpT,
        coef_delta_x,
        -coef_delta_x,
        coef_delta_y,
        -coef_delta_y
    ], axis=1)
    b_le = np.zeros(extra.m + extra.n)
    A_eq = np.concatenate([
        np.ones((1, len_w + len_z)),
        np.zeros((1, 4))
    ], axis=1)
    b_eq = np.ones(1)
    c = np.concatenate([
        np.zeros(len_w + len_z),
        [1, -1, 1, -1]
    ])
    vec_wz = []
    error_idx_w = []
    error_idx_z = []
    for i in range(extra.m):
        if i not in idx_w:
            if w[i] > EPS:
                error_idx_w.append(i)
        else:
            vec_wz.append(rho * w[i])
    for i in range(extra.n):
        if i not in idx_z:
            if z[i] > EPS:
                error_idx_z.append(i)
        else:
            vec_wz.append((1 - rho) * z[i])
    if len(error_idx_w + error_idx_z) != 0:
        return False, "error index w: {}, z: {}".format(error_idx_w, error_idx_z)
    vec_wz = np.array(vec_wz).flatten()
    dual_sol = solve_lp(A_le=A_le, b_le=b_le, A_eq=A_eq,
                        b_eq=b_eq, c=c, object_func='max')
    v_wz = np.min(G_tmpT @ np.array(vec_wz).flatten())
    # print("wz diff\n", dual_sol.x[:len_w + len_z] - vec_wz)
    if v_wz + EPS < dual_sol.fun:
        return False, "value error real {} v_wz {} with {} function\n solution {}\n wz is {} {}".format(dual_sol.fun, v_wz, "max", dual_sol.x, w.flatten(), z.flatten())
    extra.update_wz(w.reshape(-1, 1), z.reshape(-1, 1))
    res_value = {
        "ts": ts_adjust(R, C, x, y, extra)['approximation'],
        "linear": linear_adjust(R, C, x, y, extra)['approximation'],
    }

    return True, res_value


if __name__ == '__main__':
    # test tight instance of DFM
    from env import matrixgame
    from env.gen.randommatrixgenerator import RandomMatrixGenerator
    import argparse
    args = argparse.Namespace()
    eps = 1e-6
    R = np.array([[0, 0, 0], [1/3, 1, 1], [1/3, 2/3-eps/2, 2/3-eps/2]])
    C = np.array([[0, 1/3-eps, 1/3-eps], [0, 1, 2/3+eps], [0, 1, 2/3+eps]])
    args.players = 2
    args.actionspace = [R.shape[0], R.shape[1]]
    gen = RandomMatrixGenerator(args)
    game = matrixgame.MatrixGame(gen)
    init_x = np.array([1.0, 0, 0])
    init_y = np.array([1.0, 0, 0])
    args.init_x = init_x
    args.init_y = init_y
    p = Player(args=args)
    x, y = p.solve(game, [R, C])
    print(x, y)
    print(calculate_f_value(R, C, x[0], x[1]))
    R, C = C.transpose(), R.transpose()
    x, y = p.solve(game, [R, C])
    print(x, y)
    print(calculate_f_value(R, C, x[0], x[1]))
