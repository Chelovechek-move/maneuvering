import math
import numpy as np
import pytest

from maneuvering.utils.math_tools import normalize_angle
from maneuvering.orbit.convert_kep import (
    calc_mean_from_eccentric,
    calc_eccentric_from_true,
    calc_eccentric_from_mean,
    calc_true_from_eccentric,
    calc_true_from_mean,
    calc_mean_from_true,
    calc_true_from_mean_non_norm,
)

TWO_PI = 2.0 * math.pi
EPS = np.finfo(float).eps
tol = EPS * 100


def ang_diff(a, b):
    """Минимальная круговая разница углов |a-b| с приводом к [0, 2π)."""
    d = (a - b) % TWO_PI
    if d > math.pi:
        d -= TWO_PI
    return abs(d)


# ----------------------
# ГРАНИЧНЫЕ СЛУЧАИ e=0
# ----------------------

@pytest.mark.parametrize("x", [0.0, 1e-12, 1e-9, 0.1, 1.0, math.pi - 1e-12, math.pi, 2 * math.pi - 1e-12])
def test_circular_orbit_identities(x):
    """При e=0: ν == E == M (с точностью нормализации)."""
    e = 0.0
    nu = normalize_angle(x)
    E = calc_eccentric_from_true(nu, e)
    assert ang_diff(E, nu) < tol

    M = calc_mean_from_eccentric(E, e)
    assert ang_diff(M, nu) < tol

    # M -> E -> ν
    E2 = calc_eccentric_from_mean(M, e)
    nu2 = calc_true_from_eccentric(E2, e)
    assert ang_diff(E2, nu) < tol
    assert ang_diff(nu2, nu) < tol

    # ν -> M -> ν
    M2 = calc_mean_from_true(nu, e)
    nu3 = calc_true_from_mean(M2, e)
    assert ang_diff(M2, nu) < tol
    assert ang_diff(nu3, nu) < tol


# ---------------------------------------
# СЕТКИ ДЛЯ ЭЛЛИПТИЧЕСКОГО СЛУЧАЯ (e < 1)
# ---------------------------------------

@pytest.mark.parametrize("e", [0.001, 0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("nu", np.linspace(0.0, 2 * math.pi, 25, endpoint=False))
def test_true_ecc_elliptic(e, nu):
    """Эллипс: ν -> E -> ν."""
    E = calc_eccentric_from_true(nu, e)
    nu2 = calc_true_from_eccentric(E, e)
    assert ang_diff(nu2, normalize_angle(nu)) < tol


@pytest.mark.parametrize("e", [0.001, 0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("E", np.linspace(0.0, 2 * math.pi, 25, endpoint=False))
def test_ecc_mean_elliptic(e, E):
    """Эллипс: E -> M -> E (Ньютон)."""
    M = calc_mean_from_eccentric(E, e)
    E2 = calc_eccentric_from_mean(M, e)
    # сравнение именно E, а не только угловое — но приводим оба в [0,2π)
    assert ang_diff(normalize_angle(E2), normalize_angle(E)) < tol


@pytest.mark.parametrize("e", [0.001, 0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("nu", np.linspace(0.0, 2 * math.pi, 25, endpoint=False))
def test_true_mean_elliptic(e, nu):
    """Эллипс: ν -> M -> ν (через E)."""
    M = calc_mean_from_true(nu, e)
    nu2 = calc_true_from_mean(M, e)
    assert ang_diff(nu2, normalize_angle(nu)) < tol


@pytest.mark.parametrize("e", [0.001, 0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("M", np.linspace(0.0, 2 * math.pi, 25, endpoint=False))
def test_mean_true_elliptic(e, M):
    """Эллипс: M -> ν -> M."""
    nu = calc_true_from_mean(M, e)
    M2 = calc_mean_from_true(nu, e)
    assert ang_diff(M2, normalize_angle(M)) < tol


# -----------------------------
# НЕНОРМАЛИЗОВАННЫЙ ВАРИАНТ M->ν
# -----------------------------

@pytest.mark.parametrize("e", [0.0, 0.1, 0.7, 0.9])
@pytest.mark.parametrize("k", [-3, -1, 0, 1, 2, 5])
@pytest.mark.parametrize("frac", [0.0, 0.1, 1.0, math.pi - 1e-6, 2.0])
def test_true_from_mean_non_norm_preserves_whole_part(e, k, frac):
    """
    Проверяем, что calc_true_from_mean_non_norm сохраняет «целую часть»:
    пусть M = 2π*k + frac (frac в [0,2π) не обяз.)
    тогда ν_non_norm ≈ 2π*k + ν(frac), где ν(frac) = normalize(true_from_mean(frac)).
    """
    M = 2.0 * math.pi * k + frac
    nu_non_norm = calc_true_from_mean_non_norm(M, e)

    # ожидаемое значение:
    nu_frac = calc_true_from_mean(normalize_angle(frac), e)
    expected = 2.0 * math.pi * k + nu_frac

    assert abs(nu_non_norm - expected) < tol
