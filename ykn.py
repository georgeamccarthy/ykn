
# Backwards compatibility with python 2.
from __future__ import division
from __future__ import absolute_import

def y(params, gammam, gammacs, gammaself, gammae, givenYT=None, givenYc=None, debug=False):
    from numpy import zeros, arange
    # Save computation time by giving YT & Yc if you have already computed them.
    if givenYT is None:
        YT = yt(params, gammam, gammacs)
    else:
        YT = givenYT
    if givenYc is None:
        Yc = yc(params, gammam, gammacs, gammaself, YT)
    else:
        Yc = givenYc

    # Return float if input gammas are a single data point of type float.
    if isinstance(gammam, float):
        len_t = 1
        six_shape = (6, len_t)
        y_shape = len_t
    # Return 1d array if input gammas are 1d array.
    elif isinstance(gammam, ndarray):
        if gammam.ndim == 1:
            len_t = len(gammam)
            six_shape = (6, len_t)
            y_shape = len_t
        # Return 2d array if input gammas are 2d array.
        elif gammam.ndim == 2:
            len_t = gammam.shape[0]
            len_q = gammam.shape[1]
            six_shape = (6, len_t, len_q)
            y_shape = (len_t, len_q)
            YT = array([YT for i in arange(len_q)]).transpose()

    p = params.p

    Y = zeros(six_shape)
    Y_rules = zeros(six_shape)
    Y_valid = zeros(six_shape)
    Y_result = zeros(y_shape)

    gammamhat = get_gammahat(gammaself, gammam)
    gammacshat = get_gammahat(gammaself, gammacs)
    gammac = gammacs / (1 + Yc)
    gammachat = get_gammahat(gammaself, gammac)

    Y[0] = YT
    Y[1] = YT * (gammae / gammamhat) ** (-1 / 2)
    Y[2] = (YT * (gammac / gammam) * (gammae / gammachat)
        ** (-4 / 3))
    Y[3] = YT
    Y[4] = YT * (gammae / gammachat) ** ((p - 3) / 2)
    Y[5] = (
        YT
        * (gammamhat / gammachat) ** ((p - 3) / 2)
        * (gammae / gammamhat) ** (-4 / 3)
    )

    Y_rules[0] = (
    (gammac < gammam)
    & (gammae < gammamhat)
    )
    Y_rules[1] = (
        (gammac < gammam)
        & (gammamhat < gammae)
        & (gammae < gammachat)
    )
    Y_rules[2] = (
        (gammac < gammam)
        & (gammachat < gammae)
    )
    Y_rules[3] = (
        (gammam < gammac)
        & (gammae < gammachat)
    )
    Y_rules[4] = (
        (gammam < gammac)
        & (gammachat < gammae)
        & (gammae < gammamhat)

    )
    Y_rules[5] = (
        (gammam < gammac)
        & (gammamhat < gammae)
    )

    for i in arange(6):
        Y_valid[i] = Y[i] * Y_rules[i]

    Y_result = sum(Y_valid)

    if debug == True:
        return (
            Y_result,
            Y,
            Y_valid,
            Y_rules,
            Yc_result,
            Yc,
            Yc_valid,
            Yc_rules,
            gammac,
            gammachat
        )

    return Y_result

def yc(params, gammam, gammacs, gamma_self, givenYT=None, debug=False):
    from numpy import zeros, arange, ndarray, array
    # Save computation time by giving YT if you have already computed it.
    if givenYT is None:
        YT = yt(params, gammam, gammacs)
    else:
        YT = givenYT

    e_e = params.e_e
    e_b = params.e_b
    p = params.p

    gammamhat = get_gammahat(gamma_self, gammam)
    gammacshat = get_gammahat(gamma_self, gammacs)

    # Return float if input gammas are a single data point of type float.
    if isinstance(gammam, float):
        len_t = 1
        nineshape = (9, len_t)
        yc_shape = len_t
    # Return 1d array if input gammas are 1d array.
    elif isinstance(gammam, ndarray):
        if gammam.ndim == 1:
            len_t = len(gammam)
            nineshape = (9, len_t)
            yc_shape = len_t
        # Return 2d array if input gammas are 2d array.
        elif gammam.ndim == 2:
            len_t = gammam.shape[0]
            len_q = gammam.shape[1]
            nineshape = (9, len_t, len_q)
            yc_shape = (len_t, len_q)
            #YT = array([YT for i in arange(len_q)]).transpose()

    Yc = zeros(nineshape)
    Yc_valid = zeros(nineshape)
    gammac = zeros(nineshape)
    gammachat = zeros(nineshape)
    Yc_rules = zeros(nineshape)
    Yc_result = zeros(yc_shape)

    # Compute Yc in each functional regime.
    Yc[0] = YT
    Yc[1] = YT ** 2 * (gammacs / gammamhat) ** -1
    Yc[2] = YT * (gammacs / gammamhat) ** (-1 / 2)
    Yc[3] = YT * gammacs ** -1 * gammamhat ** (1 / 2)
    Yc[4] = YT
    inner_term = (
        e_e / e_b
        * 1 / (3 - p)
        * (gammam / gammacs) ** (p - 2)
        * (gammacs / gammacshat) ** ((p - 3) / 2)
    )
    Yc[5] = inner_term ** (2 / (p - 1))
    Yc[6] = inner_term
    inner_term = (
        e_e / e_b
        * 1 / (3 - p)
        * (gammam / gammamhat) ** (-4 / 3)
        * (gammam / gammacshat) ** (7 / 3)
    )
    Yc[7] = inner_term ** (3 / 7)
    Yc[8] = inner_term

    # For each Yc compute the corresponding gammac and gammachat
    for i in arange(len(Yc)):
        gammac[i] = gammacs / (1 + Yc[i])
        gammachat[i] = get_gammahat(gamma_self, gammac[i])

    # Yc_rules = 1 where each Yc obeys its own rules and = 0 where it does not.
    Yc_rules[0] = (
        (gammac[0] < gammam)
        & (gammac[0] < gammamhat)
    )
    Yc_rules[1] = (
        (gammac[1] < gammam)
        & (gammamhat < gammac[1])
        & (gammac[1] < gammachat[1])
        & (Yc[1] >= 1)
    )
    Yc_rules[2] = (
        (gammac[2] < gammam)
        & (gammamhat < gammac[2])
        & (gammac[2] < gammachat[2])
        & (Yc[2] < 1)
    )
    Yc_rules[3] = (
        (gammac[3] < gammam)
        & (gammachat[3] < gammac[3])
    )
    Yc_rules[4] = (
        (gammam < gammac[4])
        & (gammac[4] < gammachat[4])
    )
    Yc_rules[5] = (
        (gammam < gammac[5])
        & (gammachat[5] < gammac[5])
        & (gammac[5] < gammamhat)
        & (Yc[5] >= 1)
    )
    Yc_rules[6] = (
        (gammam < gammac[6])
        & (gammachat[6] < gammac[6])
        & (gammac[6] < gammamhat)
        & (Yc[6] < 1)
    )
    Yc_rules[7] = (
        (gammam < gammac[7])
        & (gammachat[7] < gammamhat)
        & (gammamhat < gammac[7])
        & (Yc[7] >= 1)
    )
    Yc_rules[8] = (
        (gammam < gammac[8])
        & (gammachat[8] < gammamhat)
        & (gammamhat < gammac[8])
        & (Yc[8] < 1)
    )
    # Remove any overlaps.
    # YT takes priority as their are fewer approximations for YT.
    Yc_rules[5][Yc_rules[4] == 1] = 0
    Yc_rules[1][Yc_rules[1] == (Yc_rules[0] == 1)] = 0

    for i in arange(9):
        Yc_valid[i] = Yc[i] * Yc_rules[i]
        Yc_result = Yc_result + Yc_valid[i]
    # FIXME fixes small gaps between valid regions at late times, giving
    # a valid Y there. Breaks early times, ultra fast cooling not valid.
    Yc_result = Yc_result + (Yc_result == 0) * YT
    if debug == False:
        return Yc_result

    # TODO remove gaps by setting blanks to YT if it's the nearset valid Yc.
    '''
    zeros = (Yc_result == 0)
    for i in arange(len_t):
        if zeros[i] == 1:
            z_index = i
            if i != 0:
                for j in arange(len_t):
                    for k in arange(9):
                        if Yc_rules[k][z_index - j] != 0:
                            lower_nonz_case = k
                            break
                    break
            else:
                lower_nonz_case = 999
            if i != len_t:
                for j in arange(len_t):
                    for k in arange(9):
                        if z_index + j >= len_t:
                            pass
                        elif Yc_rules[k][z_index + j] != 0:
                            upper_nonz_case = k
                            break
            else:
                upper_nonz_case = 999
            #if lower_nonz_case == 0 or lower_nonz_case == 4 or upper_nonz_case == 0 or upper_nonz_case == 4:
            if upper_nonz_case == 4:
                    print("test")
                    Yc_result[z_index] = YT[z_index]
    '''

    if debug == True:
        gammacvalid = zeros(shape=(9, len_t))
        for i in arange(9):
            gammacvalid[i] = gammac[i] * Yc_rules[i]
        return (Yc_result, Yc, Yc_valid, Yc_rules, gammac, gammachat, gammacvalid)

    return Yc_result


def get_gammahat(gamma_self, gamma):
    return gamma_self ** 3 / gamma ** 2

def yt(params, gammam, gammacs):
    p = params.p
    # Alpha as seen in JBH Eq.13 for smoothing.
    a = -60 * p ** -2
    return (
        YT_fast(params, gammam, gammacs) ** a
        + YT_slow(params, gammam, gammacs) ** a
    ) ** (1 / a)


# Solves A7 by passing coeffs of A7 to cubic_formula()
def YT_fast(params, gammam, gammacs):
    p = params.p
    E_ratio = params.e_e / params.e_b
    gammacsover_m = gammacs / gammam + 0j
    a = 1
    b = 2 - (p - 1) / p * gammacsover_m
    c = 1 - E_ratio - (p - 1) / p * gammacsover_m
    d = E_ratio * ((p - 2) / (p - 1) * gammacsover_m - 1)
    return cubic_formula(a, b, c, d)


# Cubic formula from applying Cardano's method to a general cubic of
# coefficients ax**3 + bx**2 + cx + d = 0.
def cubic_formula(a, b, c, d):
    solution = 0 + 0j
    A = -(b ** 3) / (27 * a ** 3) + b * c / (6 * a ** 2) - d / (2 * a)
    B = c / (3 * a) - b ** 2 / (9 * a ** 2)
    solution = (
        ((A + (A ** 2 + B ** 3) ** (1 / 2)) ** (1 / 3))
        + ((A - (A ** 2 + B ** 3) ** (1 / 2)) ** (1 / 3))
        - b / (3 * a)
    )
    return solution.real


# Computes Y Thomson in the slow regime by smoothing between the approximations
# in JBH Tab.2.
def YT_slow(params, gammam, gammacs):
    p = params.p
    # FIXME YMMV with this smoothing constant. 
    # Works well for JBH Fig.1 & Fig.1 parameters.
    a = - 1.7
    return (YT_slow_approx(params, gammam, gammacs, 2) ** a +
            YT_slow_approx(params, gammam, gammacs, 3) ** a) ** (1 / a)


# Returns an approximation for Y_slow as given in table 2 of JBH.
# t_2_row is the row number of the approximation in the table.
def YT_slow_approx(params, gammam, gammacs, t_2_row):
    p = params.p
    E_ratio = params.e_e / params.e_b
    inner_term = E_ratio / (3 - p) * (gammam / gammacs) ** (p - 2)
    # Analytic solution gives approximation for large Y.
    if t_2_row == 2:
        return inner_term ** (1 / (4 - p))
    # Analytic solution gives approximation for small Y.
    elif t_2_row == 3:
        return inner_term

# Y* as given in JBH A11
def YT_transition(params):
    p = params.p
    E_ratio = params.e_e / params.e_b
    return ((1 + 4 * p / (p - 1) * E_ratio) ** (1 / 2) - 1) / 2

def fs_transtime(params, gammam, gammacs, t):
    from numpy import where
    YTfast = YT_fast(params, gammam, gammacs)
    valid_slow = where(YTfast < YT_transition(params))
    transtime_index = max(valid_slow[0])
    return t[transtime_index]
