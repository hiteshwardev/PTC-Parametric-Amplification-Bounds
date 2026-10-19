"""Dispersive photonic time crystals: a single Lorentz oscillator with
modulated resonance or plasma frequency.

Per wavevector k (c = 1), with E = (D - P)/eps_b and Q = dP/dt,

    D' = -i k H,   H' = -i k E,   P' = Q,
    Q' = -gamma Q - wr2(t) P + wp2(t) E.

wr2 and wp2 are the squared resonance and plasma frequencies. In the harmonic
problems studied here one of them is modulated as p(t) = p0 (1 + m cos Om t)
("channel" 'wr2' or 'wp2') and the background permittivity eps_b is 1.

The two-mode theory of the first instability tongue gives the net exponent

    mu_net = |F| m w / 4 - gamma K,

with participation factors that reduce exactly to
    F_wr2 = B wr2 / Delta,  F_wp2 = A B / Delta,  K = B w^2 / (2 Delta),
where A = wr2 - w^2, B = wp2 and Delta = A^2 + B wr2. On the two branches
w_- < w_+ these are F_wr2 = (w_+^2 - wr2)/(w_+^2 - w_-^2) on the lower and
(wr2 - w_-^2)/(w_+^2 - w_-^2) on the upper branch, and |F_wp2| = wp2/(w_+^2 - w_-^2)
on both.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

WR2 = 1.0      # squared resonance frequency of the reference medium
WP2 = 0.64     # squared plasma frequency of the reference medium
RTOL, ATOL = 1e-11, 1e-13


# ---------------------------------------------------------------------------
# Static polaritons and participation factors
# ---------------------------------------------------------------------------
def polariton_omega(k, branch="lower", wr2=WR2, wp2=WP2):
    """Lossless polariton frequency solving w^2 eps(w) = k^2."""
    b = wr2 + wp2 + k * k
    root = np.sqrt(b * b - 4.0 * k * k * wr2)
    sign = -1.0 if branch == "lower" else 1.0
    return np.sqrt(0.5 * (b + sign * root))


def participation(w, wr2=WR2, wp2=WP2):
    """Exact participation factors at branch frequency w.

    Returns a dict with A, B, Delta, the G-norm N, F_wr2, F_wp2 and K.
    F_wp2 is negative on the upper branch, where A < 0.
    """
    A = wr2 - w * w
    B = wp2
    Delta = A * A + B * wr2
    return {"A": A, "B": B, "Delta": Delta, "N": 2.0 * Delta / (A + B) ** 2,
            "F_wr2": B * wr2 / Delta, "F_wp2": A * B / Delta,
            "K": B * w * w / (2.0 * Delta)}


def net_rate(k, m, gamma, channel, branch="lower", wr2=WR2, wp2=WP2):
    """Two-mode net exponent at the tongue center Om = 2 w(k).

    Returns (mu_net, w, F, K) with F signed.
    """
    w = polariton_omega(k, branch, wr2, wp2)
    pf = participation(w, wr2, wp2)
    F = pf["F_" + channel]
    return abs(F) * m * w / 4.0 - gamma * pf["K"], w, F, pf["K"]


def threshold(w, channel, gamma, wr2=WR2):
    """Modulation index at which mu_net changes sign."""
    if channel == "wr2":
        return 2.0 * gamma * w / wr2
    return 2.0 * gamma * w / abs(wr2 - w * w)


def interbranch_crossings(wr2=WR2, wp2=WP2):
    """Wavevectors where tongue tracking Om = 2 w_- meets the beat Om = w_+ - w_-.

    The two conditions coincide where w_+ = 3 w_-. With w_+ w_- = k sqrt(wr2)
    and w_+^2 + w_-^2 = wr2 + wp2 + k^2 this gives
    k^2 - (10/3) sqrt(wr2) k + wr2 + wp2 = 0.
    """
    b = 10.0 / 3.0 * math.sqrt(wr2)
    disc = b * b - 4.0 * (wr2 + wp2)
    if disc < 0:
        return []
    return sorted([0.5 * (b - math.sqrt(disc)), 0.5 * (b + math.sqrt(disc))])


# ---------------------------------------------------------------------------
# Floquet monodromy under harmonic modulation of one channel
# ---------------------------------------------------------------------------
def monodromy(k, gamma, m, Om, channel, wr2=WR2, wp2=WP2,
              rtol=RTOL, atol=ATOL):
    """4x4 monodromy with all four columns integrated in one pass."""
    T = 2.0 * math.pi / Om

    def rhs(t, y):
        Y = y.reshape(4, 4)
        c = 1.0 + m * math.cos(Om * t)
        a = wr2 * c if channel == "wr2" else wr2
        b = wp2 * c if channel == "wp2" else wp2
        E = Y[0] - Y[2]
        return np.concatenate([(-1j * k) * Y[1], (-1j * k) * E, Y[3],
                               -gamma * Y[3] - a * Y[2] + b * E])

    sol = solve_ivp(rhs, (0.0, T), np.eye(4, dtype=complex).reshape(-1),
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(4, 4), T


def tongue_pair(M, T):
    """Exponents of the multiplier pair closest to the negative real axis.

    At the first tongue the parametrically coupled pair has quasi-frequency
    Om/2, i.e. multipliers near -1. Returns (mu_pair, split, mu_max), where
    split is the difference of the two pair exponents (zero outside the gap).
    """
    ev = np.linalg.eigvals(M)
    mu = np.log(np.abs(ev)) / T
    pair = np.argsort(np.abs(np.abs(np.angle(ev)) - math.pi))[:2]
    return (float(max(mu[pair])), float(abs(mu[pair[0]] - mu[pair[1]])),
            float(mu.max()))


def liouville_error(M, gamma, T):
    """|det M - exp(-gamma T)|; the trace of the generator is -gamma."""
    return float(abs(np.linalg.det(M) - math.exp(-gamma * T)))


# ---------------------------------------------------------------------------
# General time dependence (benchmarks and energy balance)
# ---------------------------------------------------------------------------
class LorentzMedium:
    """Lorentz medium with arbitrary periodic wr2(t), wp2(t) and eps_b(t)."""

    def __init__(self, k, gamma=0.0, Om=2.0, wr2=None, wp2=None, epsb=None):
        self.k, self.gamma, self.Om = float(k), float(gamma), float(Om)
        self.T = 2.0 * math.pi / self.Om
        self.wr2 = wr2 if wr2 is not None else (lambda t: WR2 + 0.0 * np.asarray(t))
        self.wp2 = wp2 if wp2 is not None else (lambda t: WP2 + 0.0 * np.asarray(t))
        self.epsb = epsb if epsb is not None else (lambda t: 1.0 + 0.0 * np.asarray(t))

    def rhs(self, t, u):
        D, H, P, Q = u
        E = (D - P) / self.epsb(t)
        return [-1j * self.k * H, -1j * self.k * E, Q,
                -self.gamma * Q - self.wr2(t) * P + self.wp2(t) * E]

    def monodromy(self):
        cols = []
        for j in range(4):
            sol = solve_ivp(self.rhs, (0.0, self.T), np.eye(4, dtype=complex)[:, j],
                            method="DOP853", rtol=RTOL, atol=ATOL)
            cols.append(sol.y[:, -1])
        return np.array(cols).T

    def max_exponent(self):
        ev = np.linalg.eigvals(self.monodromy())
        return float(np.max(np.log(np.abs(ev))) / self.T)


# ---------------------------------------------------------------------------
# Growth inequality along trajectories
# ---------------------------------------------------------------------------
def growth_inequality_violation(k, gamma, Om, mr=0.0, mp=0.0, phase=0.0,
                                u0=(1.0, 0.3 + 0.2j, 0.1, 0.05j), periods=2,
                                n_out=2001, wr2=WR2, wp2=WP2, rtol=RTOL, atol=ATOL):
    """Largest violation of the integrated growth inequality on one trajectory.

    The resonance and plasma frequencies are modulated as
    wr2(t) = wr2 (1 + mr cos Om t) and wp2(t) = wp2 (1 + mp cos(Om t + phase)).
    Along the trajectory the right-hand side

        R = f_pot |d ln wr2/dt| + f_m |d ln wp2/dt| - 2 gamma f_kin

    is integrated together with the field, and g(t) = int_0^t R - ln[W(t)/W(0)]
    must be non-decreasing. The function returns max over t_i <= t_j of
    g(t_i) - g(t_j), which is zero up to round-off when the inequality holds on
    every sub-interval of the sampled trajectory.
    """
    def params(t):
        cr, sr = math.cos(Om * t), math.sin(Om * t)
        cp, sp = math.cos(Om * t + phase), math.sin(Om * t + phase)
        a, b = wr2 * (1.0 + mr * cr), wp2 * (1.0 + mp * cp)
        dla = -mr * Om * sr / (1.0 + mr * cr)
        dlb = -mp * Om * sp / (1.0 + mp * cp)
        return a, b, dla, dlb

    def energy_parts(y, a, b):
        D, H, P, Q = y[:4]
        E = D - P
        field = 0.5 * (abs(E) ** 2 + abs(H) ** 2)
        pot = a * abs(P) ** 2 / (2.0 * b)
        kin = abs(Q) ** 2 / (2.0 * b)
        return field, pot, kin

    def rhs(t, y):
        D, H, P, Q, _ = y
        a, b, dla, dlb = params(t)
        E = D - P
        field, pot, kin = energy_parts(y, a, b)
        W = field + pot + kin
        R = (pot * abs(dla) + (pot + kin) * abs(dlb) - 2.0 * gamma * kin) / W
        return [-1j * k * H, -1j * k * E, Q, -gamma * Q - a * P + b * E, R]

    # |d ln wr2/dt| and |d ln wp2/dt| have kinks where sin vanishes; the
    # integration is restarted there so that every segment is smooth.
    T = 2.0 * math.pi / Om
    t_end = periods * T
    kinks = set()
    for active, shift in ((mr != 0.0, 0.0), (mp != 0.0, phase)):
        if active:
            n0 = math.ceil(shift / math.pi)
            for n in range(n0, n0 + 2 * periods + 2):
                t = (n * math.pi - shift) / Om
                if 0.0 < t < t_end:
                    kinks.add(t)
    edges = [0.0] + sorted(kinks) + [t_end]
    tt = np.linspace(0.0, t_end, n_out)
    states = np.empty((5, n_out), dtype=complex)
    y = np.concatenate([np.asarray(u0, dtype=complex), [0.0]])
    states[:, 0] = y
    for a, b in zip(edges[:-1], edges[1:]):
        inside = np.nonzero((tt > a) & (tt <= b))[0]
        sol = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=rtol, atol=atol,
                        t_eval=np.unique(np.append(tt[inside], b)))
        if not sol.success:
            raise RuntimeError(sol.message)
        states[:, inside] = sol.y[:, :len(inside)]
        y = sol.y[:, -1]
    lnW = np.empty(n_out)
    for j, t in enumerate(tt):
        a, b, _, _ = params(t)
        lnW[j] = math.log(sum(energy_parts(states[:, j], a, b)))
    g = np.real(states[4]) - (lnW - lnW[0])
    return float(np.max(np.maximum.accumulate(g) - g))


def static_roots(k, gamma, wr2=WR2, wp2=WP2):
    """Roots s = -i w of (s^2 + k^2)(s^2 + gamma s + wr2 + wp2) - k^2 wp2 = 0."""
    return np.roots([1.0, gamma, wr2 + wp2 + k * k, gamma * k * k, k * k * wr2])


def multiset_distance(a, b):
    """Largest distance between greedily matched elements of a and b."""
    b = list(b)
    worst = 0.0
    for x in a:
        d = [abs(x - y) for y in b]
        j = int(np.argmin(d))
        worst = max(worst, d[j])
        b.pop(j)
    return worst


# ---------------------------------------------------------------------------
# Four-mode secular theory at an interbranch crossing
# ---------------------------------------------------------------------------
def four_mode_matrix(k, gamma, m, channel, wr2=WR2, wp2=WP2):
    """Secular matrix of the set {+w_-, -w_-, +w_+, -w_+} at Om = 2 w_-.

    Near w_+ = 3 w_- the degenerate tongue pair couples to the upper branch
    through the same-sign beats (+w_-, +w_+) and (-w_-, -w_+). The matrix is
    built by projecting the modulation and the loss with the energy form G on
    the four eigenvectors, in a frame rotating at w_- for the lower modes and
    at 3 w_- for the upper ones. The real parts of its eigenvalues approximate
    the four Floquet exponents of the medium.
    """
    G = np.array([[1, 0, -1, 0], [0, 1, 0, 0],
                  [-1, 0, 1 + wr2 / wp2, 0], [0, 0, 0, 1 / wp2]], dtype=complex)
    A1 = np.zeros((4, 4), dtype=complex)
    if channel == "wr2":
        A1[3, 2] = -wr2
    else:
        A1[3, 0], A1[3, 2] = wp2, -wp2

    wl = polariton_omega(k, "lower", wr2, wp2)
    wu = polariton_omega(k, "upper", wr2, wp2)
    vec = {}
    for name, w, s in (("a", wl, 1), ("b", wl, -1), ("c", wu, 1), ("d", wu, -1)):
        p = 1.0 - w * w / (k * k)
        vec[name] = np.array([1.0, s * w / k, p, -1j * s * w * p], dtype=complex)
    norm = {n: float(np.real(np.conj(v) @ G @ v)) for n, v in vec.items()}
    kappa = {"a": gamma * participation(wl, wr2, wp2)["K"],
             "c": gamma * participation(wu, wr2, wp2)["K"]}
    kappa["b"], kappa["d"] = kappa["a"], kappa["c"]

    def c(l, j):
        return (m / 2.0) * (np.conj(vec[l]) @ G @ A1 @ vec[j]) / norm[l]

    detune = wu - 3.0 * wl
    S = np.array([[-kappa["a"], c("a", "b"), c("a", "c"), 0],
                  [c("b", "a"), -kappa["b"], 0, c("b", "d")],
                  [c("c", "a"), 0, -kappa["c"] - 1j * detune, 0],
                  [0, c("d", "b"), 0, -kappa["d"] + 1j * detune]], dtype=complex)
    return S
