
from __future__ import division
from __future__ import absolute_import
def yc(params, gamma_m, gamma_cs, gamma_self, givenYT=None, debug=False):
    from numpy import zeros, arange
    # Save computation time by giving YT if you have already computed it.
    if givenYT is None:
        YT = yt(params, gamma_m, gamma_cs)
    else:
        YT = givenYT

    e_e = params.e_e
    e_b = params.e_b
    p = params.p

    gamma_mhat = get_gammahat(gamma_self, gamma_m)
    gamma_cshat = get_gammahat(gamma_self, gamma_cs)

    # Make compatible with t as float
    if isinstance(gamma_m, float):
        len_t = 1
    else:
        len_t = len(gamma_m)

    Yc = zeros(shape=(9, len_t))
    Yc_valid = zeros(shape=(9,len_t))
    gamma_c = zeros(shape=(9, len_t))
    gamma_chat = zeros(shape=(9, len_t))
    if debug == True:
        gammacvalid = zeros(shape=(9, len_t))
    Yc_rules = zeros(shape=(9, len_t))
    Yc_result = zeros(len_t)

    # Compute Yc in each functional regime.
    Yc[0] = YT
    Yc[1] = YT ** 2 * (gamma_cs / gamma_mhat) ** -1
    Yc[2] = YT * (gamma_cs / gamma_mhat) ** (-1 / 2)
    Yc[3] = YT * gamma_cs ** -1 * gamma_mhat ** (1 / 2)
    Yc[4] = YT
    inner_term = (
        e_e
        / (e_b * (3 - p))
        * (gamma_m / gamma_cs) ** (p - 2)
        * (gamma_cs / gamma_cshat) ** ((p - 3) / 2)
    )
    Yc[5] = inner_term ** (2 / (p - 1))
    Yc[6] = inner_term
    inner_term = (
        e_e
        / (e_b * (3 - p))
        * (gamma_m / gamma_mhat) ** (-4 / 3)
        * (gamma_m / gamma_cshat) ** (7 / 3)
    )
    Yc[7] = inner_term ** (3 / 7)
    Yc[8] = inner_term

    # For each Yc compute the corresponding gammac and gammachat
    for i in arange(len(Yc)):
        gamma_c[i] = gamma_cs / (1 + Yc[i])
        gamma_chat[i] = get_gammahat(gamma_self, gamma_c[i])

    # Yc_rules = 1 where each Yc obeys its own rules and = 0 where it does not.
    Yc_rules[0] = (
        (gamma_c[0] < gamma_m)
        & (gamma_c[0] < gamma_mhat)
    )
    Yc_rules[1] = (
        (gamma_c[1] < gamma_m)
        & (gamma_mhat < gamma_c[1])
        & (gamma_c[1] < gamma_chat[1])
        & (Yc[1] >= 1)
    )
    Yc_rules[2] = (
        (gamma_c[2] < gamma_m)
        & (gamma_mhat < gamma_c[2])
        & (gamma_c[2] < gamma_chat[2])
        & (Yc[2] < 1)
    )
    Yc_rules[3] = (
        (gamma_c[3] < gamma_m)
        & (gamma_chat[3] < gamma_c[3])
    )
    Yc_rules[4] = (
        (gamma_m < gamma_c[4])
        & (gamma_c[4] < gamma_chat[4])
    )
    Yc_rules[5] = (
        (gamma_m < gamma_c[5])
        & (gamma_chat[5] < gamma_c[5])
        & (gamma_c[5] < gamma_mhat)
        & (Yc[5] >= 1)
    )
    Yc_rules[6] = (
        (gamma_m < gamma_c[6])
        & (gamma_chat[6] < gamma_c[6])
        & (gamma_c[6] < gamma_mhat)
        & (Yc[6] < 1)
    )
    Yc_rules[7] = (
        (gamma_m < gamma_c[7])
        & (gamma_chat[7] < gamma_mhat)
        & (gamma_mhat < gamma_c[7])
        & (Yc[7] >= 1)
    )
    Yc_rules[8] = (
        (gamma_m < gamma_c[8])
        & (gamma_chat[8] < gamma_mhat)
        & (gamma_mhat < gamma_c[8])
        & (Yc[8] < 1)
    )
    # Remove any overlaps.
    # YT takes priority as their are fewer approximations for YT.
    Yc_rules[5][Yc_rules[4] == 1] = 0
    Yc_rules[1][Yc_rules[1] == (Yc_rules[0] == 1)] = 0

    for i in arange(9):
        Yc_valid[i] = Yc[i] * Yc_rules[i]
        Yc_result = Yc_result + Yc_valid[i]

    if debug == True:
        for i in arange(9):
            gammacvalid[i] = gamma_c[i] * Yc_rules[i]
        return (Yc_result, Yc, Yc_valid, Yc_rules, gamma_c, gamma_chat, gammacvalid)

    return Yc_result


def get_gammahat(gamma_self, gamma):
    return gamma_self ** 3 / gamma ** 2

def yt(params, gamma_m, gamma_cs):
    p = params.p
    # Alpha as seen in JBH Eq.13 for smoothing.
    a = -60 * p ** -2
    return (
        YT_fast(params, gamma_m, gamma_cs) ** a
        + YT_slow(params, gamma_m, gamma_cs) ** a
    ) ** (1 / a)


# Solves A7 by passing coeffs of A7 to cubic_formula()
def YT_fast(params, gamma_m, gamma_cs):
    p = params.p
    E_ratio = params.e_e / params.e_b

    gamma_cs_over_m = gamma_cs / gamma_m + 0j
    a = 1
    b = 2 - (p - 1) / p * gamma_cs_over_m
    c = 1 - E_ratio - (p - 1) / p * gamma_cs_over_m
    d = E_ratio * ((p - 2) / (p - 1) * gamma_cs_over_m - 1)
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
def YT_slow(params, gamma_m, gamma_cs):
    p = params.p
    # FIXME YMMV with this smoothing constant. 
    # Works well for JBH Fig.1 & Fig.1 parameters.
    a = - 1.7
    return (YT_slow_approx(params, gamma_m, gamma_cs, 2) ** a +
            YT_slow_approx(params, gamma_m, gamma_cs, 3) ** a) ** (1 / a)


# Returns an approximation for Y_slow as given in table 2 of JBH.
# t_2_row is the row number of the approximation in the table.
def YT_slow_approx(params, gamma_m, gamma_cs, t_2_row):
    p = params.p
    E_ratio = params.e_e / params.e_b
    inner_term = E_ratio / (3 - p) * (gamma_m / gamma_cs) ** (p - 2)
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
