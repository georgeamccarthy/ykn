# Backwards compatibility with python 2.
from __future__ import division
from __future__ import absolute_import


def y(params, gammam, gammac, gammamhat, gammachat, gammae, YT, debug=False):
    from numpy import zeros, arange, ndarray, array

    # Return float if input gammas are a single data point of type float.
    if isinstance(gammae, float) or isinstance(gammae, int):
        dims = 0
        # For a 1D array, dimension can be time of frequency.
        len_tq = 1
        six_shape = (6, len_tq)
        y_shape = len_tq
    # Return 1d array if input gammas are 1d array.
    elif isinstance(gammae, ndarray):
        if gammae.ndim == 1:
            dims = 1
            # If 1d array, we don't know if time of freq space.
            len_tq = len(gammae)
            six_shape = (6, len_tq)
            y_shape = len_tq
        # Return 2d array if input gammas are 2d array.
        elif gammae.ndim == 2:
            dims = 2
            len_t = gammam.shape[0]
            len_q = gammam.shape[1]
            six_shape = (6, len_t, len_q)
            y_shape = (len_t, len_q)

    p = params.p

    Y = zeros(six_shape)
    Y_rules = zeros(six_shape)
    Y_valid = zeros(six_shape)
    Y_result = zeros(y_shape)

    # Compute Y for 6 orderings of the critical lorentz factors.
    Y[0] = YT
    Y[1] = YT * (gammae / gammamhat) ** (-1 / 2)
    Y[2] = YT * (gammac / gammam) * (gammae / gammachat) ** (-4 / 3)
    Y[3] = YT
    Y[4] = YT * (gammae / gammachat) ** ((p - 3) / 2)
    Y[5] = (
        YT * (gammamhat / gammachat) ** ((p - 3) / 2) * (gammae / gammamhat) ** (-4 / 3)
    )

    # Compute boundaries for each of the 6 Y parameters.
    # Y_rules[i] = 1 where Y[i] has its conditions on the critical
    # lorentz factors satisfied.
    Y_rules[0] = (gammac < gammam) & (gammae < gammamhat)
    Y_rules[1] = (gammac < gammam) & (gammamhat < gammae) & (gammae < gammachat)
    Y_rules[2] = (gammac < gammam) & (gammachat < gammae)
    Y_rules[3] = (gammam < gammac) & (gammae < gammachat)
    Y_rules[4] = (gammam < gammac) & (gammachat < gammae) & (gammae < gammamhat)
    Y_rules[5] = (gammam < gammac) & (gammamhat < gammae)

    for i in arange(6):
        Y_valid[i] = Y[i] * Y_rules[i]

    Y_result = sum(Y_valid)

    # Returns extra information for plotting, diagnostics and debugging if
    # debug parameter is true.
    if debug == True:
        return (Y_result, Y, Y_valid, Y_rules, gammac, gammachat)

    return Y_result


def yc_approx(params, gammam, gammacs, gamma_self, YT=None, debug=False):
    from numpy import zeros, arange, ndarray, array

    e_e = params.e_e
    e_b = params.e_b
    p = params.p

    # Return float if input gammas are a single data point of type float.
    if isinstance(gammam, float) or isinstance(gammam, int):
        dims = 0
        len_t = 1
    # Return 1d array if input gammas are 1d array.
    elif isinstance(gammam, ndarray):
        if gammam.ndim == 1:
            # Results in Yc returned as 1D array.
            dims = 1
            len_t = len(gammam)
        # Return 2d array if input gammas are 2d array.
        elif gammam.ndim == 2:
            # Results in Yc returned as 2D array.
            dims = 2
            len_t = gammam.shape[0]
            len_q = gammam.shape[1]
            # Slice gammas to 1D arrays because for Yc they are degenerate
            # in frequency.
            gammam = gammam[:, 0]
            gammacs = gammacs[:, 0]
            gamma_self = gamma_self[:, 0]
            if YT is not None and YT.ndim == 2:
                YT = YT[:, 0]
    else:
        raise Warning("Unsupported types for one or more gamma arguments.")

    # Save computation time by giving YT if you have already computed it.
    if YT is None:
        YT = yt(params, gammam, gammacs)

    gammamhat = get_gammahat(gamma_self, gammam)
    gammacshat = get_gammahat(gamma_self, gammacs)

    nineshape = (9, len_t)
    Yc = zeros(nineshape)

    # Compute Yc in each functional regime.
    Yc[0] = YT
    Yc[1] = YT ** 2 * (gammacs / gammamhat) ** -1
    # Alternate equation
    # Yc[1] = YT ** (2/3) * (gammacs / gammamhat) ** (-1/3)
    Yc[2] = YT * (gammacs / gammamhat) ** (-1 / 2)
    Yc[3] = YT * gammacs ** -1 * gammamhat ** (1 / 2)
    Yc[4] = YT
    inner_term = (
        e_e
        / e_b
        * (p - 2)
        / (3 - p)
        * (gammam / gammacs) ** (p - 2)
        * (gammacs / gammacshat) ** ((p - 3) / 2)
    )
    Yc[5] = inner_term ** (2 / (p - 1))
    Yc[6] = inner_term
    inner_term = (
        e_e
        / e_b
        * 1
        / (3 - p)
        * (gammam / gammamhat) ** (-4 / 3)
        * (gammam / gammacshat) ** (7 / 3)
    )
    Yc[7] = inner_term ** (3 / 7)
    Yc[8] = inner_term

    # gammac, gammachat across all 9 regimes in two broadcast ops.
    gammac    = gammacs / (1 + Yc)
    gammachat = get_gammahat(gamma_self, gammac)

    # Combine the 9 candidate Yc values into a single Yc(t) by weighting
    # each regime by the product of soft indicators on the strict
    # inequalities that define it. YT picks up any leftover weight; a
    # final soft-min with YT keeps Yc <= YT.

    # Smoothing sharpness exponent. Matches `pl = 2` used by realspectra
    # for the time-domain GS-regime blending and by knspectrum for KN
    # sub-spectrum branch selection, so the same transition width applies
    # everywhere a smoothed `a < b` shows up in the pipeline.
    pl = 2

    def _less_than(a, b):
        # Smooth indicator for `a < b`: ~1 when a << b, ~0 when a >> b,
        # 0.5 at a = b. The form 1 / (1 + (a/b)**pl) is the Granot &
        # Sari (2002) sigmoid.
        return 1.0 / (1.0 + (a / b) ** pl)

    one = 1.0
    w_smooth = zeros(nineshape)
    w_smooth[0] = (_less_than(gammac[0], gammam)
                   * _less_than(gammac[0], gammamhat))
    w_smooth[1] = (_less_than(gammac[1], gammam)
                   * _less_than(gammamhat, gammac[1])
                   * _less_than(gammac[1], gammachat[1])
                   * _less_than(one, Yc[1]))
    w_smooth[2] = (_less_than(gammac[2], gammam)
                   * _less_than(gammamhat, gammac[2])
                   * _less_than(gammac[2], gammachat[2])
                   * _less_than(Yc[2], one))
    w_smooth[3] = (_less_than(gammac[3], gammam)
                   * _less_than(gammachat[3], gammac[3]))
    w_smooth[4] = (_less_than(gammam, gammac[4])
                   * _less_than(gammac[4], gammachat[4])
                   * _less_than(Yc[4], one))
    w_smooth[5] = (_less_than(gammam, gammac[5])
                   * _less_than(gammachat[5], gammac[5])
                   * _less_than(gammac[5], gammamhat)
                   * _less_than(one, Yc[5]))
    w_smooth[6] = (_less_than(gammam, gammac[6])
                   * _less_than(gammachat[6], gammac[6])
                   * _less_than(gammac[6], gammamhat)
                   * _less_than(Yc[6], one))
    w_smooth[7] = (_less_than(gammam, gammac[7])
                   * _less_than(gammachat[7], gammamhat)
                   * _less_than(gammamhat, gammac[7])
                   * _less_than(one, Yc[7]))
    w_smooth[8] = (_less_than(gammam, gammac[8])
                   * _less_than(gammachat[8], gammamhat)
                   * _less_than(gammamhat, gammac[8])
                   * _less_than(Yc[8], one))

    # sum along axis 0 collapses the 9 regimes into per-t totals;
    # w_sum[t] is the total weight assigned at time t. Normalising by
    # w_sum gives the regimes' weighted average Yc. YT is added only as
    # a strict gap-filler — `gap` becomes ~1 only when w_sum drops to
    # zero (no regime fits), so well-covered points get the regime
    # average without YT bias. Without the gap test, a few percent of
    # leftover weight times a huge YT would dominate the answer wherever
    # YT >> Yc.
    w_sum = w_smooth.sum(axis=0)
    gap = _less_than(w_sum, 0.01)
    denom = w_sum + gap
    Yc_result = ((w_smooth * Yc).sum(axis=0) + gap * YT) / denom

    # Soft-min cap to enforce Yc <= YT. The combiner
    #   (x^a + y^a)^(1/a)
    # approaches min(x, y) for a -> -infinity and matches it everywhere
    # outside a narrow window around x = y. a = -60/p^2 is the smoothing
    # exponent Jacovich, Beniamini & van der Horst (JBH) Eq. 13 use to
    # blend the fast- and slow-cooling YT branches; reusing it here keeps
    # the cap's transition width consistent with the rest of yt().
    a_cap = -60.0 / p ** 2
    Yc_result = (Yc_result ** a_cap + YT ** a_cap) ** (1.0 / a_cap)

    if debug == False:
        if dims == 2:
            return array([Yc_result for i in arange(len_q)]).transpose()
        return Yc_result

    # debug path needs the hard-rule arrays for diagnostics
    Yc_rules = zeros(nineshape)
    Yc_rules[0] = (gammac[0] < gammam) & (gammac[0] < gammamhat)
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
    Yc_rules[3] = (gammac[3] < gammam) & (gammachat[3] < gammac[3])
    Yc_rules[4] = (gammam < gammac[4]) & (gammac[4] < gammachat[4]) & (Yc[4] < 1)
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
    Yc_valid = Yc * Yc_rules

    if debug == True:
        gammacvalid = zeros(shape=(9, len_t))
        for i in arange(9):
            gammacvalid[i] = gammac[i] * Yc_rules[i]
        return (Yc_result, Yc, Yc_valid, Yc_rules, gammac, gammachat, gammacvalid)


# Compute gammahat.
def get_gammahat(gamma_self, gamma):
    return gamma_self ** 3 / gamma ** 2


# Compute Y Thomson given p, gammam and gammacs.
def yt(params, gammam, gammacs):
    from numpy import ndarray, array, arange

    p = params.p

    dims = 0
    if isinstance(gammam, ndarray):
        if gammam.ndim == 2:
            dims = 2
            len_t = gammam.shape[0]
            len_q = gammam.shape[1]
            gammam = gammam[:, 0]
            gammacs = gammacs[:, 0]

    # Alpha as seen in JBH Eq.13 for smoothing.
    a = -60 * p ** -2

    YT = (
        YT_fast(params, gammam, gammacs) ** a + YT_slow(params, gammam, gammacs) ** a
    ) ** (1 / a)

    if dims == 2:
        return array([YT for i in arange(len_q)]).transpose()

    return YT


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
        (A + (A ** 2 + B ** 3) ** (1 / 2)) ** (1 / 3)
        + (A - (A ** 2 + B ** 3) ** (1 / 2)) ** (1 / 3)
        - b / (3 * a)
    )
    return solution.real


# Computes Y Thomson in the slow regime by smoothing between the approximations
# in JBH Tab.2.
def YT_slow(params, gammam, gammacs):
    p = params.p
    # FIXME YMMV with this smoothing constant.
    # Works well for JBH Fig.1 & Fig.1 parameters.
    a = -1.7
    return (
        YT_slow_approx(params, gammam, gammacs, 2) ** a
        + YT_slow_approx(params, gammam, gammacs, 3) ** a
    ) ** (1 / a)


# Returns an approximation for Y_slow as given in table 2 of JBH.
# t_2_row is the row number of the approximation in the table.
def YT_slow_approx(params, gammam, gammacs, t_2_row):
    p = params.p
    E_ratio = params.e_e / params.e_b
    inner_term = E_ratio / (3 - p) * (gammam / gammacs) ** (p - 2)
    # print(inner_term)
    # Analytic solution gives approximation for large Y.
    if t_2_row == 2:
        # print(1/(4-p))
        # print(inner_term ** (1 / (4 - p)))
        return inner_term ** (1 / (4 - p))
    # Analytic solution gives approximation for small Y.
    elif t_2_row == 3:
        return inner_term


# JBH A15 solved numerically for exact YT_slow.
def YT_slow_exact(params, gammam, gammacs):
    from numpy import arange, zeros
    from scipy.optimize import fsolve

    p = params.p
    E_ratio = params.e_e / params.e_b
    YT = zeros(len(gammam))

    def A15(YT):
        return YT * (1 + YT) ** 2 * (
            p * (1 + YT) ** (1 - p) - gamma_m_over_cs ** (p - 1)
        ) - p * E_ratio * (
            gamma_m_over_cs * (1 + YT) ** (3 - p) * (p - 2) / (p - 3)
            + 1 / (3 - p) * gamma_m_over_cs ** (p - 2)
        )

    for i in arange(len(gammam)):
        starting_guess = YT_slow_approx(params, gammam[i], gammacs[i], 2)
        gamma_m_over_cs = float(gammam[i] / gammacs[i])
        YT[i] = float(fsolve(A15, starting_guess))

    return YT


# Y* as given in JBH A11
def YT_transition(params):
    p = params.p
    E_ratio = params.e_e / params.e_b
    return ((1 + 4 * p / (p - 1) * E_ratio) ** (1 / 2) - 1) / 2


# Computes the transition time between fast and slow regimes in the Thomson
# regime. Useful for plotting.
def fs_transtime(params, gammam, gammacs, t):
    from numpy import where

    YTfast = YT_fast(params, gammam, gammacs)
    valid_slow = where(YTfast < YT_transition(params))
    transtime_index = max(valid_slow[0])
    return t[transtime_index]
