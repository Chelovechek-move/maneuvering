import math

import numpy as np

from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.maneuvers.quasi_circular.transition.execute import execute, execute_batch
from maneuvering.orbit.keplerian import KepTrue

TWO_PI = 2.0 * math.pi
mu = 3.9860044158e14  # [м^3/с^2]


def test_execute_no_maneuvers():
    """Если манёвров нет, орбита не меняется."""
    a = 7000e3
    o0 = KepTrue(a=a, e=0.0, w=0.0, i=0.0, raan=0.0, nu=1.234)
    o1 = execute(o0, maneuvers=[], mu=mu)

    assert o1.a == o0.a
    assert o1.e == o0.e
    assert o1.i == o0.i
    assert o1.raan == o0.raan
    assert o1.w == o0.w
    assert o1.nu == o0.nu


def test_transversal_impulse_at_nu0_near_circular():
    """
    Круговая орбита (e=0), чисто тангенциальный импульс при ν=0.
    Ожидаем:
      Δa ≈ 2 a Δv_t / V
      Δe ≈ 2 Δv_t / V
    """
    a = 7000e3
    e = 0.0
    V = math.sqrt(mu / a)  # орбитальная скорость на круговой
    dv_t = 0.1  # [м/с], малый импульс в трансверсальном направлении (ось t)
    oi = KepTrue(a=a, e=e, w=0.0, i=0.0, raan=0.0, nu=0.0)

    # манёвр задан истинной широтой u = w + nu; здесь u=0
    m = Maneuver(dv=np.array([0.0, dv_t, 0.0], dtype=np.float64), angle=0.0)
    o1 = execute(oi, [m], mu)

    da_expected = 2.0 * a * (dv_t / V)
    de_expected = 2.0 * (dv_t / V)

    # Проверяем аппроксимацию околокруговой теории
    assert math.isclose(o1.a - a, da_expected, rel_tol=1e-4)
    assert math.isclose(o1.e - e, de_expected, rel_tol=1e-5)


def test_execute_batch_no_maneuvers():
    """Если манёвров нет, орбита не меняется."""
    a = 7000e3
    step = np.deg2rad(3)
    o0 = KepTrue(a=a, e=0.0, w=0.0, i=0.0, raan=0.0, nu=1.234)
    batch = execute_batch(o0, maneuvers=[], step=step, mu=mu)

    assert len(batch) == 1
    o1 = batch[0]
    assert o1.a == o0.a
    assert o1.e == o0.e
    assert o1.i == o0.i
    assert o1.raan == o0.raan
    assert o1.w == o0.w
    assert o1.nu == o0.nu


def test_execute_comparison():
    """Сравнение двух функций прогноза движения по финальной точке"""
    a = 7000e3
    step = np.deg2rad(3)
    o0 = KepTrue(a=a, e=0.0, w=0.0, i=0.0, raan=0.0, nu=0.0)
    m1 = Maneuver(dv=np.array([0.1, 1, 0.1], dtype=np.float64), angle=0.1)
    m2 = Maneuver(dv=np.array([0.1, 1, 0.1], dtype=np.float64), angle=0.2)
    batch = execute_batch(o0, maneuvers=[m1, m2], step=step, mu=mu)
    of1 = batch[-1]

    of2 = execute(o0, maneuvers=[m1, m2], mu=mu)

    assert of1.a == of2.a
    assert of1.e == of2.e
    assert of1.i == of2.i
    assert of1.raan == of2.raan
    assert of1.w == of2.w
    assert of1.nu == of2.nu
