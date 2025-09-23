import math

import numpy as np

from maneuvering.orbit.convert_kep import (
    calc_mean_from_true,
    calc_true_from_mean,
)
from maneuvering.orbit.keplerian import KepMean, KepTrue
from maneuvering.orbit.propagate import (
    propagate_mean,
    propagate_mean_anomaly,
    propagate_true,
    propagate_true_anomaly,
)
from maneuvering.orbit.utils import mean_motion, period

mu = 3.9860044158e14
TWO_PI = 2.0 * math.pi
EPS = np.finfo(float).eps
tol = EPS * 100


def ang_diff(a: float, b: float) -> float:
    """Минимальная круговая разница углов |a-b| с приведением к (-π, π]."""
    d = (a - b) % TWO_PI
    if d > math.pi:
        d -= TWO_PI
    return abs(d)


def test_propagate_mean_anomaly_linear():
    """M(t+dt) = M + n·dt — линейность по времени."""
    a = 7000e3
    n = mean_motion(a, mu)
    M0 = 0.123
    for dt in [0.0, 1.0, 10.0, 1234.5, -200.0]:
        M1 = propagate_mean_anomaly(M0, a, dt, mu)
        assert M1 == M0 + n * dt


def test_propagate_mean_anomaly_one_period():
    """Через полный период средняя аномалия увеличивается ровно на 2π."""
    a = 7200e3
    T = period(a, mu)
    M0 = 2.2
    M1 = propagate_mean_anomaly(M0, a, T, mu)
    assert math.isclose(M1 - M0, TWO_PI, rel_tol=0.0, abs_tol=tol)


def test_propagate_mean_dataclass_updates_only_M():
    """propagate_mean изменяет только M, остальное неизменно."""
    a = 6800e3
    o0 = KepMean(a=a, e=0.01, w=0.3, i=0.2, raan=1.1, M=0.9)
    o1 = propagate_mean(o0, dt=100.0, mu=mu)
    assert o1.a == o0.a and o1.e == o0.e and o1.w == o0.w and o1.i == o0.i and o1.raan == o0.raan
    assert o1.M != o0.M


def test_true_half_period_equivalence_to_mean_plus_pi():
    """
    Через полпериода: ν(t+T/2) == true_from_mean( M(ν) + π ).
    Точность — жёсткая (итеративный решатель, но это базовый сценарий).
    """
    a = 6700e3
    e = 0.1
    n = mean_motion(a, mu)
    T = TWO_PI / n

    # несколько стартовых ν
    for nu0 in np.linspace(0.0, TWO_PI, 9, endpoint=False):
        # путь через истинную
        nu_true = propagate_true_anomaly(nu0, a, e, dt=T / 2.0, mu=mu)

        # эквивалент через среднюю
        M0 = calc_mean_from_true(nu0, e)
        M_half = M0 + math.pi
        nu_ref = calc_true_from_mean(M_half, e)

        assert ang_diff(nu_true, nu_ref) < tol


def test_true_full_period_adds_2pi_k():
    """Через k полных периодов истинная аномалия увеличивается на 2πk (с сохранением «целой части»)."""
    a = 7000e3
    e = 0.3
    n = mean_motion(a, mu)
    T = TWO_PI / n
    nu0 = 1.234

    for k in [-2, -1, 0, 1, 2, 5]:
        dt = k * T
        nu1 = propagate_true_anomaly(nu0, a, e, dt, mu)
        expected = nu0 + k * TWO_PI
        # Здесь сравниваем напрямую, без модуло — функция сохраняет целую часть.
        assert math.isclose(nu1, expected, rel_tol=0.0, abs_tol=tol)


def test_propagate_true_vs_mean_consistency_small_dt():
    """
    Малый шаг: propagate_true ≈ путь ν→M (через calc_mean_from_true), затем M'→ν'.
    Сравниваем, что дают одинаковый результат.
    """
    a = 7050e3
    e = 0.12
    dt = 0.123  # малый шаг
    for nu0 in np.linspace(0.0, TWO_PI, 7, endpoint=False):
        # прямой способ
        nu1 = propagate_true_anomaly(nu0, a, e, dt, mu)
        # через среднюю
        M0 = calc_mean_from_true(nu0, e)
        M1 = propagate_mean_anomaly(M0, a, dt, mu)
        nu_ref = calc_true_from_mean(M1, e)
        assert ang_diff(nu1, nu_ref) < tol


def test_circular_orbit_e0_true_equals_mean_everywhere():
    """
    Для e=0 истинная и средняя аномалии совпадают всегда.
    Проверяем propagate_true и propagate_mean на совпадение.
    """
    a = 7200e3
    e = 0.0

    for M0 in np.linspace(0.0, TWO_PI, 7, endpoint=False):
        nu0 = M0  # e=0 → ν==M
        for dt in [0.0, 1.0, 10.0, 1234.0, -50.0]:
            # Через mean
            M1 = propagate_mean_anomaly(M0, a, dt, mu)
            # Через true
            nu1 = propagate_true_anomaly(nu0, a, e, dt, mu)
            # Сравниваем углы с учётом круговой топологии
            assert ang_diff(nu1 % TWO_PI, M1 % TWO_PI) < tol


def test_propagate_true_dataclass_updates_only_nu():
    """propagate_true изменяет только nu, остальное неизменно."""
    o0 = KepTrue(a=6800e3, e=0.05, w=0.4, i=0.1, raan=1.2, nu=0.7)
    o1 = propagate_true(o0, dt=500.0, mu=mu)
    assert o1.a == o0.a and o1.e == o0.e and o1.w == o0.w and o1.i == o0.i and o1.raan == o0.raan
    assert o1.nu != o0.nu


def test_propagate_true_and_mean_cross_check():
    """
    Кросс-проверка: возьмём KepTrue и соответствующий KepMean (согласованные ν и M),
    распротянем оба на dt и сверим согласованность M'↔ν'.
    """
    a = 7000e3
    e = 0.25
    for nu0 in np.linspace(0.0, TWO_PI, 9, endpoint=False):
        M0 = calc_mean_from_true(nu0, e)
        kt = KepTrue(a=a, e=e, w=0.0, i=0.0, raan=0.0, nu=nu0)
        km = KepMean(a=a, e=e, w=0.0, i=0.0, raan=0.0, M=M0)
        dt = 321.0

        kt1 = propagate_true(kt, dt, mu)
        km1 = propagate_mean(km, dt, mu)

        # Средняя из ν' должна совпасть с M' (мод 2π).
        M_from_nu1 = calc_mean_from_true(kt1.nu, e)
        assert ang_diff(M_from_nu1 % TWO_PI, km1.M % TWO_PI) < tol
