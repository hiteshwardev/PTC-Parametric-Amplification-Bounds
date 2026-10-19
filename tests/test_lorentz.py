import math

import numpy as np
import pytest

import lorentz as L

MEDIA = [0.64, 1.0, 2.25, 4.0, 9.0]


def eigenvector_form(k, w, wr2, wp2):
    # Participation factors from the on-shell eigenvector v = (1, w/k, p, -i w p),
    # e = w^2/k^2, p = 1 - e, with G-norm N = e^2 + e + (wr2 + w^2) p^2 / wp2.
    e = w * w / (k * k)
    p = 1.0 - e
    N = e * e + e + (wr2 + w * w) * p * p / wp2
    return {"N": N, "F_wr2": 2.0 * wr2 * p * p / (wp2 * N),
            "F_wp2": 2.0 * e * p / N, "K": w * w * p * p / (wp2 * N)}


@pytest.mark.parametrize("wp2", MEDIA)
@pytest.mark.parametrize("branch", ["lower", "upper"])
def test_reduced_factors_equal_eigenvector_form(wp2, branch):
    for k in np.linspace(0.1, 5.0, 25):
        w = L.polariton_omega(k, branch, 1.0, wp2)
        red = L.participation(w, 1.0, wp2)
        eig = eigenvector_form(k, w, 1.0, wp2)
        for key in ("N", "F_wr2", "F_wp2", "K"):
            assert abs(red[key] - eig[key]) <= 1e-10 * abs(eig[key])


@pytest.mark.parametrize("wp2", MEDIA)
def test_unconditional_ceiling(wp2):
    for branch in ("lower", "upper"):
        w = L.polariton_omega(np.linspace(0.05, 6.0, 200), branch, 1.0, wp2)
        pf = L.participation(w, 1.0, wp2)
        assert np.all(np.abs(pf["F_wr2"]) < 1.0)
        assert np.all(np.abs(pf["F_wp2"]) < 1.0)


def test_thresholds_are_zeros_of_the_net_rate():
    for channel in ("wr2", "wp2"):
        for k in (0.5, 1.2, 2.5):
            w = L.polariton_omega(k)
            mc = L.threshold(w, channel, 0.01)
            mu, _, _, _ = L.net_rate(k, mc, 0.01, channel)
            assert abs(mu) < 1e-15


def test_crossings_satisfy_three_to_one_ratio():
    for k in L.interbranch_crossings():
        ratio = L.polariton_omega(k, "upper") / L.polariton_omega(k, "lower")
        assert abs(ratio - 3.0) < 1e-12


def test_static_multipliers_match_dispersion_roots():
    for gamma in (0.0, 0.02):
        med = L.LorentzMedium(k=1.3, gamma=gamma, Om=2.0)
        ev = np.linalg.eigvals(med.monodromy())
        th = np.exp(L.static_roots(1.3, gamma) * med.T)
        assert L.multiset_distance(ev, th) < 1e-8


def test_stacked_monodromy_matches_reference_and_liouville():
    k, gamma, m = 0.8, 0.01, 0.05
    Om = 2.0 * L.polariton_omega(k)
    M, T = L.monodromy(k, gamma, m, Om, "wp2")
    assert L.liouville_error(M, gamma, T) < 1e-10
    ref = L.LorentzMedium(k=k, gamma=gamma, Om=Om,
                          wp2=lambda t: L.WP2 * (1.0 + m * np.cos(Om * np.asarray(t))))
    mu = np.max(np.log(np.abs(np.linalg.eigvals(M)))) / T
    assert abs(mu - ref.max_exponent()) < 1e-12


def test_two_mode_rate_against_floquet():
    k, gamma, m = 1.5, 0.01, 0.02
    for channel in ("wr2", "wp2"):
        pred, w, _, _ = L.net_rate(k, m, gamma, channel)
        M, T = L.monodromy(k, gamma, m, 2.0 * w, channel)
        mu_pair, split, _ = L.tongue_pair(M, T)
        assert split > 1e-9
        assert abs(mu_pair - pred) < 0.01 * abs(pred)


def test_four_mode_rate_reduces_to_two_mode_far_from_crossing():
    for channel in ("wr2", "wp2"):
        two, _, _, _ = L.net_rate(1.5, 0.05, 0.01, channel)
        four = np.max(np.linalg.eigvals(L.four_mode_matrix(1.5, 0.01, 0.05, channel)).real)
        assert abs(four - two) < 0.02 * abs(two)


def test_four_mode_spectrum_at_a_crossing():
    k = L.interbranch_crossings()[0]
    w = L.polariton_omega(k)
    M, T = L.monodromy(k, 0.01, 0.05, 2.0 * w, "wp2")
    exact = np.sort(np.log(np.abs(np.linalg.eigvals(M))) / T)[::-1]
    four = np.sort(np.linalg.eigvals(L.four_mode_matrix(k, 0.01, 0.05, "wp2")).real)[::-1]
    assert np.max(np.abs(exact - four)) < 1e-4


def test_growth_inequality_holds_along_a_trajectory():
    w = L.polariton_omega(1.5)
    for kw in (dict(mr=0.1), dict(mp=0.1), dict(mr=0.05, mp=0.05, phase=1.0)):
        assert L.growth_inequality_violation(1.5, 0.03, 2.0 * w, **kw) < 1e-9


def test_upper_branch_lies_above_longitudinal_frequency():
    w = L.polariton_omega(np.linspace(0.0, 4.0, 50), "upper")
    assert np.all(w ** 2 >= L.WR2 + L.WP2 - 1e-12)
    assert math.isclose(L.polariton_omega(0.0, "upper") ** 2, L.WR2 + L.WP2)


@pytest.mark.parametrize("wp2", MEDIA)
def test_branch_forms_of_the_participation_factors(wp2):
    k = np.linspace(0.05, 6.0, 120)
    wl, wu = L.polariton_omega(k, "lower", 1.0, wp2), L.polariton_omega(k, "upper", 1.0, wp2)
    lo, up = L.participation(wl, 1.0, wp2), L.participation(wu, 1.0, wp2)
    split = wu ** 2 - wl ** 2
    assert np.allclose(lo["F_wr2"] + up["F_wr2"], 1.0, rtol=0, atol=1e-12)
    assert np.allclose(lo["F_wr2"], (wu ** 2 - 1.0) / split, rtol=1e-11)
    assert np.allclose(np.abs(lo["F_wp2"]), wp2 / split, rtol=1e-11)
    assert np.allclose(np.abs(up["F_wp2"]), wp2 / split, rtol=1e-11)
    assert np.allclose(lo["K"], wl ** 2 * lo["F_wr2"] / 2.0, rtol=1e-11)


@pytest.mark.parametrize("wp2", MEDIA)
def test_largest_plasma_channel_participation(wp2):
    k = np.linspace(0.0, 6.0, 60001)
    F = np.abs(L.participation(L.polariton_omega(k, "lower", 1.0, wp2), 1.0, wp2)["F_wp2"])
    wp0 = math.sqrt(wp2)
    closed = wp0 / 2.0 if wp0 <= 1.0 else wp2 / (1.0 + wp2)
    assert abs(F.max() - closed) < 1e-8


def test_beat_coupling_sign_depends_on_channel():
    for k in L.interbranch_crossings():
        for channel, sign in (("wr2", -1.0), ("wp2", 1.0)):
            S = L.four_mode_matrix(k, 0.01, 0.05, channel)
            assert np.sign((S[0, 2] * S[2, 0]).real) == sign
            assert (S[0, 1] * S[1, 0]).real > 0.0
