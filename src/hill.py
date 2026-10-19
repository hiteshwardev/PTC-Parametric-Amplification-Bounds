"""Nondispersive photonic time crystals: Hill equation, Floquet exponents and
the total-variation bound.

For a plane wave of wavevector k in a medium with periodic permittivity
eps(t), the displacement field obeys (c = 1)

    D'' + w(t)^2 D = 0,        w(t)^2 = k^2 / eps(t).

The amplitude exponent mu of the largest Floquet multiplier satisfies

    mu T <= TV_T[ln eps] / 4,

where TV_T is the total variation over one period T. The same bound holds for
the norm of the monodromy matrix measured in the Hill-energy norm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

TWO_PI = 2.0 * math.pi
RTOL, ATOL = 1e-11, 1e-13


# ---------------------------------------------------------------------------
# Modulation waveforms. Each class provides eps(t), d(ln eps)/dt, the period T
# and a step cap small enough to resolve its fastest feature.
# ---------------------------------------------------------------------------
@dataclass
class LogFourierWaveform:
    """eps(t) = ebar exp(sum_n a_n cos(n Om t + phi_n))."""
    a: np.ndarray
    phi: np.ndarray
    Om: float = 1.0
    ebar: float = 1.0

    @property
    def T(self) -> float:
        return TWO_PI / self.Om

    def _phase(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        n = np.arange(1, len(self.a) + 1)
        return np.outer(n, t) * self.Om + self.phi[:, None], n

    def eps(self, t):
        scalar = np.ndim(t) == 0
        ph, _ = self._phase(t)
        out = self.ebar * np.exp(np.sum(self.a[:, None] * np.cos(ph), axis=0))
        return float(out[0]) if scalar else out

    def dlneps_dt(self, t):
        scalar = np.ndim(t) == 0
        ph, n = self._phase(t)
        out = np.sum(-self.a[:, None] * (n[:, None] * self.Om) * np.sin(ph), axis=0)
        return float(out[0]) if scalar else out

    def depth(self) -> float:
        """Permittivity contrast max(eps)/min(eps)."""
        e = self.eps(np.linspace(0.0, self.T, 4096, endpoint=False))
        return float(e.max() / e.min())

    def max_step(self) -> float:
        return self.T / (20.0 * max(1, len(self.a)))

    def scalar_eps(self):
        terms = [(float(a), (i + 1) * self.Om, float(p))
                 for i, (a, p) in enumerate(zip(self.a, self.phi))]
        ebar = self.ebar

        def eps(t):
            s = 0.0
            for a, nOm, p in terms:
                s += a * math.cos(nOm * t + p)
            return ebar * math.exp(s)
        return eps


@dataclass
class SinusoidWaveform:
    """eps(t) = ebar (1 + delta cos(Om t))."""
    delta: float
    Om: float = 1.0
    ebar: float = 1.0

    @property
    def T(self) -> float:
        return TWO_PI / self.Om

    def eps(self, t):
        return self.ebar * (1.0 + self.delta * np.cos(self.Om * np.asarray(t, float)))

    def dlneps_dt(self, t):
        t = np.asarray(t, dtype=float)
        return (-self.delta * self.Om * np.sin(self.Om * t)
                / (1.0 + self.delta * np.cos(self.Om * t)))

    def depth(self) -> float:
        return (1.0 + self.delta) / (1.0 - self.delta)

    def max_step(self) -> float:
        return self.T / 20.0

    def scalar_eps(self):
        d, Om, ebar = self.delta, self.Om, self.ebar
        return lambda t: ebar * (1.0 + d * math.cos(Om * t))


@dataclass
class TwoLevelWaveform:
    """Smoothed two-level switching between eps1 and eps2.

    ln eps rises from ln eps1 to ln eps2 at phase p1 and falls back at phase p2
    (both as fractions of T) through tanh ramps of width w (fraction of T).
    As w -> 0 the profile approaches sudden switching.
    """
    eps1: float
    eps2: float
    p1: float
    p2: float
    w: float
    Om: float = 1.0

    @property
    def T(self) -> float:
        return TWO_PI / self.Om

    @property
    def _mid_half(self):
        l1, l2 = math.log(self.eps1), math.log(self.eps2)
        return 0.5 * (l1 + l2), 0.5 * (l2 - l1)

    def _g(self, t):
        x = (np.asarray(t, float) / self.T) % 1.0
        g = -np.ones_like(x)
        for shift in (-1.0, 0.0, 1.0):      # periodic images keep g smooth at x = 0
            g += (np.tanh((x - self.p1 + shift) / self.w)
                  - np.tanh((x - self.p2 + shift) / self.w))
        return g

    @staticmethod
    def _sech2(z):
        # sech^2 written so that large |z| underflows to zero instead of
        # overflowing cosh
        z = np.clip(np.abs(np.asarray(z, float)), 0.0, 350.0)
        q = np.exp(-2.0 * z)
        return 4.0 * q / (1.0 + q) ** 2

    def eps(self, t):
        mid, half = self._mid_half
        return np.exp(mid + half * self._g(t))

    def dlneps_dt(self, t):
        _, half = self._mid_half
        x = (np.asarray(t, float) / self.T) % 1.0
        dg = np.zeros_like(x)
        for shift in (-1.0, 0.0, 1.0):
            dg += (self._sech2((x - self.p1 + shift) / self.w)
                   - self._sech2((x - self.p2 + shift) / self.w))
        return half * dg / (self.w * self.T)

    def depth(self) -> float:
        return max(self.eps1, self.eps2) / min(self.eps1, self.eps2)

    def max_step(self) -> float:
        return 0.5 * self.w * self.T

    def scalar_eps(self):
        T, (mid, half) = self.T, self._mid_half
        p1, p2, w = self.p1, self.p2, self.w

        def eps(t):
            x = (t / T) % 1.0
            g = -1.0
            for shift in (-1.0, 0.0, 1.0):
                g += math.tanh((x - p1 + shift) / w) - math.tanh((x - p2 + shift) / w)
            return math.exp(mid + half * g)
        return eps


def random_log_fourier(n_wf: int, seed: int, n_harm: int = 6, amp: float = 0.35,
                       depth_cap: float = 4.0, Om: float = 1.0):
    """Random log-Fourier profiles with a_n ~ U(-amp, amp)/n, phi_n ~ U(0, 2 pi).

    Draws are rejected until the permittivity contrast is at most depth_cap.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_wf:
        a = rng.uniform(-amp, amp, size=n_harm) / np.arange(1, n_harm + 1)
        phi = rng.uniform(0.0, TWO_PI, size=n_harm)
        wf = LogFourierWaveform(a=a, phi=phi, Om=Om)
        if wf.depth() <= depth_cap:
            out.append(wf)
    return out


# ---------------------------------------------------------------------------
# Total variation and the bound
# ---------------------------------------------------------------------------
def total_variation(wf, n: int = 1024) -> float:
    """TV_T[ln eps], exact up to round-off.

    ln eps is monotone between consecutive zeros of d(ln eps)/dt, so the total
    variation is the sum of |Delta ln eps| over those monotone runs. The zeros
    are bracketed on a grid, the grid is refined until the number of brackets
    no longer changes, and each zero is located with Brent's method.
    """
    def brackets(m):
        tt = np.linspace(0.0, wf.T, m + 1)
        d = np.asarray(wf.dlneps_dt(tt), dtype=float)
        return tt, np.nonzero(d[:-1] * d[1:] < 0.0)[0]

    tt, idx = brackets(n)
    for _ in range(4):
        tt2, idx2 = brackets(2 * n)
        if len(idx2) == len(idx):
            break
        n, tt, idx = 2 * n, tt2, idx2
    else:
        raise RuntimeError("turning points of ln eps not resolved")

    def deriv(t):
        return float(np.atleast_1d(wf.dlneps_dt(t))[0])

    nodes = [0.0]
    for i in idx:
        try:
            nodes.append(brentq(deriv, tt[i], tt[i + 1], xtol=1e-15))
        except ValueError:
            nodes.append(0.5 * (tt[i] + tt[i + 1]))
    nodes.append(wf.T)
    lneps = np.log(np.asarray(wf.eps(np.asarray(nodes)), dtype=float))
    return float(np.abs(np.diff(lneps)).sum())


def bound(wf) -> float:
    """Right-hand side of the bound, TV_T[ln eps] / 4."""
    return 0.25 * total_variation(wf)


# ---------------------------------------------------------------------------
# Monodromy and Floquet exponents
# ---------------------------------------------------------------------------
def monodromy(wf, k: float, rtol: float = RTOL, atol: float = ATOL,
              max_step: float | None = None) -> np.ndarray:
    """Monodromy matrix of D'' + (k^2/eps) D = 0 acting on (D, D')."""
    eps = wf.scalar_eps()
    k2 = k * k

    def rhs(t, y):
        w2 = k2 / eps(t)
        return (y[2], y[3], -w2 * y[0], -w2 * y[1])

    if max_step is None:
        max_step = wf.max_step()
    sol = solve_ivp(rhs, (0.0, wf.T), [1.0, 0.0, 0.0, 1.0], method="DOP853",
                    rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y[:, -1]
    return np.array([[y[0], y[1]], [y[2], y[3]]])


def exponent(M: np.ndarray, T: float) -> float:
    """Largest Floquet amplitude exponent mu = ln|lambda_max| / T."""
    return float(np.max(np.log(np.abs(np.linalg.eigvals(M)))) / T)


def floquet_exponent(wf, k: float, **kw) -> float:
    return exponent(monodromy(wf, k, **kw), wf.T)


def wronskian_error(M: np.ndarray) -> float:
    """|det M - 1|; the Hill flow preserves area, so det M = 1 exactly."""
    return float(abs(np.linalg.det(M) - 1.0))


def energy_norm(M: np.ndarray, wf, k: float) -> float:
    """Operator norm of M in the Hill-energy norm E = (D'^2 + w(0)^2 D^2)/2."""
    w0 = k / math.sqrt(float(np.atleast_1d(wf.eps(0.0))[0]))
    S = np.diag([w0, 1.0])
    return float(np.linalg.norm(S @ M @ np.linalg.inv(S), 2))


# ---------------------------------------------------------------------------
# Sudden two-level switching: exact transfer matrices
# ---------------------------------------------------------------------------
def _free_flight(w: float, tau: float) -> np.ndarray:
    """Propagator of D'' + w^2 D = 0 over a time tau, acting on (D, D')."""
    c, s = math.cos(w * tau), math.sin(w * tau)
    return np.array([[c, s / w], [-w * s, c]])


def sudden_monodromy(eps1: float, eps2: float, p1: float, p2: float, k: float,
                     Om: float = 1.0) -> np.ndarray:
    """Exact monodromy for eps = eps2 on [p1 T, p2 T) and eps1 elsewhere.

    D and D' are continuous across a jump of eps (D'' stays bounded), so
    the period map is a product of three free flights.
    """
    T = TWO_PI / Om
    w1, w2 = k / math.sqrt(eps1), k / math.sqrt(eps2)
    return (_free_flight(w1, (1.0 - p2) * T) @ _free_flight(w2, (p2 - p1) * T)
            @ _free_flight(w1, p1 * T))


def sudden_optimum(eps1: float, eps2: float, Om: float = 1.0):
    """Wavevector and dwell fraction at which sudden switching attains the bound.

    Each level is held for a quarter of its own oscillation period, so every
    jump of eps meets the field at a turning point of D or of D'. Returns
    (k, f2), with f2 the fraction of the period spent at eps2.
    """
    s1, s2 = math.sqrt(eps1), math.sqrt(eps2)
    return Om * (s1 + s2) / 4.0, s2 / (s1 + s2)


def energy_angle_residual(wf, k: float, y0=(0.7, 0.3)) -> float:
    """Residual of ln[E(T)/E(0)] + int (w'/w) cos(2 theta) dt over one period."""
    h = 1e-7 * wf.T

    def w_of(t):
        return k / math.sqrt(float(np.atleast_1d(wf.eps(t))[0]))

    def rhs(t, y):
        w = w_of(t)
        wdot = (w_of(t + h) - w_of(t - h)) / (2.0 * h)
        D, Dd, _ = y
        cos2theta = (Dd * Dd - w * w * D * D) / (Dd * Dd + w * w * D * D)
        return [Dd, -w * w * D, -(wdot / w) * cos2theta]

    sol = solve_ivp(rhs, (0.0, wf.T), [y0[0], y0[1], 0.0], method="DOP853",
                    rtol=RTOL, atol=ATOL)
    D, Dd, integral = sol.y[:, -1]
    E0 = 0.5 * (y0[1] ** 2 + w_of(0.0) ** 2 * y0[0] ** 2)
    ET = 0.5 * (Dd ** 2 + w_of(wf.T) ** 2 * D ** 2)
    return float(abs(math.log(ET / E0) - integral))
