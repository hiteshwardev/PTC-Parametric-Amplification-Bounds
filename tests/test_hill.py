import math

import numpy as np
import pytest

import hill


@pytest.mark.parametrize("delta", [0.05, 0.4, 0.85])
def test_total_variation_sinusoid(delta):
    wf = hill.SinusoidWaveform(delta=delta)
    exact = 2.0 * math.log((1.0 + delta) / (1.0 - delta))
    assert abs(hill.total_variation(wf) - exact) < 1e-12


def test_total_variation_single_harmonic_and_two_level():
    wf = hill.LogFourierWaveform(a=np.array([0.3]), phi=np.array([0.7]))
    assert abs(hill.total_variation(wf) - 1.2) < 1e-12
    wf = hill.TwoLevelWaveform(1.0, 3.0, 0.2, 0.7, 0.02)
    assert abs(hill.total_variation(wf) - 2.0 * math.log(3.0)) < 1e-9


def test_constant_permittivity_has_unit_multipliers():
    wf = hill.SinusoidWaveform(delta=0.0)
    for k in (0.3, 0.8, 1.7):
        assert abs(hill.floquet_exponent(wf, k)) < 1e-10


def test_mathieu_tongue_center_rate():
    wf = hill.SinusoidWaveform(delta=0.02)
    mu = max(hill.floquet_exponent(wf, k) for k in np.linspace(0.499, 0.501, 11))
    assert abs(mu - 0.02 * 0.5 / 4.0) / (0.02 * 0.5 / 4.0) < 0.03


def test_energy_angle_identity():
    wf = hill.SinusoidWaveform(delta=0.04)
    for k in (0.42, 0.5, 0.9):
        assert hill.energy_angle_residual(wf, k) < 1e-8


def test_bound_and_operator_norm_on_random_profiles():
    for wf in hill.random_log_fourier(5, seed=3):
        b = hill.bound(wf)
        for k in np.linspace(0.3, 1.8, 10):
            M = hill.monodromy(wf, k)
            log_rho = hill.exponent(M, wf.T) * wf.T
            log_norm = math.log(hill.energy_norm(M, wf, k))
            assert hill.wronskian_error(M) < 1e-9
            assert log_rho <= log_norm + 1e-12
            assert log_norm <= b + 1e-9


@pytest.mark.parametrize("ratio", [1.2, 2.0, 4.0])
def test_sudden_switching_attains_the_bound(ratio):
    k, f2 = hill.sudden_optimum(1.0, ratio)
    M = hill.sudden_monodromy(1.0, ratio, 0.1, 0.1 + f2, k)
    log_rho = math.log(np.max(np.abs(np.linalg.eigvals(M))))
    assert abs(log_rho - 0.5 * math.log(ratio)) < 1e-12
    assert hill.wronskian_error(M) < 1e-12


def test_smooth_two_level_approaches_sudden_limit():
    k, f2 = hill.sudden_optimum(1.0, 2.0)
    sat = []
    for w in (0.02, 0.01):
        wf = hill.TwoLevelWaveform(1.0, 2.0, 0.1, 0.1 + f2, w)
        sat.append(hill.floquet_exponent(wf, k) * wf.T / hill.bound(wf))
    assert sat[0] < sat[1] < 1.0
