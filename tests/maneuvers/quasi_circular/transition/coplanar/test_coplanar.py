import math

import numpy as np

from maneuvering.maneuvers.quasi_circular.reference_orbit import reference_orbit
from maneuvering.maneuvers.quasi_circular.transition.coplanar.coplanar import coplanar_analytical
from maneuvering.maneuvers.quasi_circular.transition.execute import execute
from maneuvering.orbit.distance import distance_orbit
from maneuvering.orbit.keplerian import Kep, KepTrue

DEG = math.pi / 180.0
mu = 3.9860044158e14  # [м^3/с^2]
tol = np.finfo(float).eps * 100.0

# ----------------------------------------------------------------------------------------------------------------------
# Тест из книжки Баранова
# ----------------------------------------------------------------------------------------------------------------------


def test_baranov_example():
    """Пример из Баранова: проверяем два ΔV."""

    # Начальная и целевая орбиты
    oi = KepTrue(a=6566000.0, e=0.00228, w=20.0 * DEG, i=0.0, raan=0.0, nu=0.0)
    ot = KepTrue(a=6721000.0, e=0.00149, w=150.0 * DEG, i=0.0, raan=0.0, nu=0.0)

    mans = coplanar_analytical(oi, ot, mu)

    # Эталонные значения
    delta_v1 = 51.8327
    delta_v2 = 38.5273
    tol = 1e-2

    assert len(mans) == 2
    assert math.isclose(mans[0].dv[1], delta_v1, rel_tol=0.0, abs_tol=tol)
    assert math.isclose(mans[1].dv[1], delta_v2, rel_tol=0.0, abs_tol=tol)


# ----------------------------------------------------------------------------------------------------------------------
# Тест на совпадающие орбиты
# ----------------------------------------------------------------------------------------------------------------------


def test_no_maneuvers():
    """Совпадающие орбиты → манёвров нет."""
    oi = Kep(a=7_000_000.0, e=0.00228, w=20 * DEG, i=0.0, raan=0.0)
    ot = Kep(a=7_000_000.0, e=0.00228, w=20 * DEG, i=0.0, raan=0.0)

    mans = coplanar_analytical(oi, ot, mu)
    assert len(mans) == 0


# ----------------------------------------------------------------------------------------------------------------------
# Сравнение захардкоженных значений на каждый из алгоритмов маневрирования
# ----------------------------------------------------------------------------------------------------------------------


def test_non_intersecting_case():
    """Непересекающиеся орбиты: проверяем углы, компоненты импульсов и суммарный ΔV."""
    oi = Kep(a=6_566_000.0, e=0.00228, w=20 * DEG, i=0.0, raan=130 * DEG)
    ot = Kep(a=6_721_000.0, e=0.00149, w=150 * DEG, i=0.0, raan=130 * DEG)

    mans = coplanar_analytical(oi, ot, mu)

    # Эталонные значения
    angle1 = 2.802665451883477
    imp1_r = 0.0
    imp1_t = 51.827889726288454
    imp1_n = 0.0

    angle2 = 5.944258105473271
    imp2_r = 0.0
    imp2_t = 38.53189066725772
    imp2_n = 0.0

    # Суммарная характеристическая скорость по эталонной формуле
    ref = reference_orbit(oi, ot, mu)
    da = (ot.a - oi.a) / ref.r
    dv = da / 2.0

    assert len(mans) == 2

    # Манёвр 1
    assert math.isclose(mans[0].angle, angle1, abs_tol=tol)
    assert math.isclose(float(mans[0].dv[0]), imp1_r, abs_tol=tol)
    assert math.isclose(float(mans[0].dv[1]), imp1_t, abs_tol=1e-5)
    assert math.isclose(float(mans[0].dv[2]), imp1_n, abs_tol=tol)

    # Манёвр 2
    assert math.isclose(mans[1].angle, angle2, abs_tol=tol)
    assert math.isclose(float(mans[1].dv[0]), imp2_r, abs_tol=tol)
    assert math.isclose(float(mans[1].dv[1]), imp2_t, abs_tol=1e-5)
    assert math.isclose(float(mans[1].dv[2]), imp2_n, abs_tol=tol)

    # Проверка суммы |t|-компонент с эталоном
    sum_dv = abs(float(mans[0].dv[1])) + abs(float(mans[1].dv[1]))
    assert math.isclose(dv * ref.v, sum_dv, abs_tol=tol)


def test_intersecting_case():
    """Пересекающиеся орбиты: проверяем углы, компоненты импульсов и суммарный ΔV."""
    oi = KepTrue(a=6_566_000.0, e=0.00228, w=20 * DEG, i=0.0, raan=130 * DEG, nu=0.0)
    ot = KepTrue(a=6_576_000.0, e=0.00149, w=150 * DEG, i=0.0, raan=130 * DEG, nu=0.0)

    mans = coplanar_analytical(oi, ot, mu)

    # Эталонные значения
    angle1 = 2.802665451883477
    imp1_t = 9.6477785893852062
    imp1_r = 0.0
    imp1_n = 0.0

    angle2 = 5.944258105473271
    imp2_t = -3.7213688167914718
    imp2_r = 0.0
    imp2_n = 0.0

    def ecc_vec(o: Kep) -> np.ndarray:
        return o.e * np.array([math.cos(o.w), math.sin(o.w)], dtype=np.float64)

    ref = reference_orbit(oi, ot, mu)
    de = ecc_vec(ot) - ecc_vec(oi)
    dv = float(np.linalg.norm(de)) / 2.0

    # Манёвр 1
    assert math.isclose(mans[0].angle, angle1, abs_tol=tol)
    assert math.isclose(float(mans[0].dv[0]), imp1_r, abs_tol=tol)
    assert math.isclose(float(mans[0].dv[1]), imp1_t, abs_tol=1e-5)
    assert math.isclose(float(mans[0].dv[2]), imp1_n, abs_tol=tol)

    # Манёвр 2
    assert math.isclose(mans[1].angle, angle2, abs_tol=tol)
    assert math.isclose(float(mans[1].dv[0]), imp2_r, abs_tol=tol)
    assert math.isclose(float(mans[1].dv[1]), imp2_t, abs_tol=1e-5)
    assert math.isclose(float(mans[1].dv[2]), imp2_n, abs_tol=tol)

    # Проверка суммы |t|-компонент с эталоном
    sum_dv = abs(float(mans[0].dv[1])) + abs(float(mans[1].dv[1]))
    assert math.isclose(dv * ref.v, sum_dv, abs_tol=tol)


# ----------------------------------------------------------------------------------------------------------------------
# Тесты на изменение большой полуоси
# ----------------------------------------------------------------------------------------------------------------------


def _calc_total_dv(mans) -> float:
    """Сумма норм импульсов манёвров, [м/с]."""
    s = 0.0
    for m in mans:
        s += float(np.linalg.norm(m.dv))
    return s


def _check(da_min: int, da_max: int, step: int, error_tol: float) -> None:
    """
    - генерим пары орбит с Δa в заданном диапазоне,
    - строим манёвры coplanar_analytical,
    - прогоняем execute,
    - сверяем метрику distance(final, target) и суммарный ΔV.
    """
    for i in range(da_min, da_max + 1, step):
        ai = 6800e3
        at = 6800e3 + i

        # Исходная и целeвая орбиты
        oi = KepTrue(a=ai, e=0.0, w=0.0, i=0.0, raan=0.0, nu=0.0)
        ot = KepTrue(a=at, e=0.0, w=0.0, i=0.0, raan=0.0, nu=0.0)

        # Манёвры по приближённой компланарной схеме
        mans = coplanar_analytical(oi, ot, mu)

        # Применяем манёвры
        of = execute(oi, mans, mu)

        final_distance = distance_orbit(of, ot)
        assert final_distance <= error_tol

        # Суммарная характеристическая скорость
        dv = _calc_total_dv(mans)

        # Эталон: V * Δa / (2 * r_ref), где r_ref = (ai + at)/2, V = sqrt(mu / r_ref)
        r_ref = 0.5 * (ai + at)
        V = math.sqrt(mu / r_ref)
        dv_ref = V * (at - ai) / (2.0 * r_ref)

        assert math.isclose(dv, dv_ref, rel_tol=0.0, abs_tol=abs(dv_ref) * 1e-13)


def test_transition_error_leq_1km_semi_major_axis():
    # tolerance = 0
    _check(da_min=0, da_max=1000, step=10, error_tol=0.0)


def test_transition_error_from_1km_to_10km_semi_major_axis():
    # tolerance = 1.83783
    _check(da_min=1000, da_max=10000, step=50, error_tol=1.83783)


def test_transition_error_from_10km_to_50km_semi_major_axis():
    # tolerance = 45.95756
    _check(da_min=10000, da_max=50000, step=50, error_tol=45.95756)


def test_transition_error_from_50km_to_100km_semi_major_axis():
    # tolerance = 183.82607
    _check(da_min=50000, da_max=100000, step=100, error_tol=183.82607)
