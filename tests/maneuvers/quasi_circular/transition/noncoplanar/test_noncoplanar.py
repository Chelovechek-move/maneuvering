from __future__ import annotations

import numpy as np

from maneuvering.maneuvers.quasi_circular.reference_orbit import (
    reference_orbit,
    trans_devs,
)
from maneuvering.maneuvers.quasi_circular.transition.noncoplanar.noncoplanar import (
    noncoplanar_analytical,
)
from maneuvering.orbit.keplerian import Kep

deg: float = np.pi / 180.0
tolerance: float = np.finfo(float).eps * 1e2
mu: float = 3.98600415e14


def sqr(x: float) -> float:
    return x * x


def assert_vec_close(vec, x, y, z, atol):
    assert np.isfinite(vec).all()
    assert np.isclose(vec[0], x, atol=atol)
    assert np.isclose(vec[1], y, atol=atol)
    assert np.isclose(vec[2], z, atol=atol)


def test_nodal_case_without_ex():
    # Узловой случай с нулевой проекцией вектора эксцентриситета на ось Ox
    initial = Kep(7_000_000, 0.00228, 90 * deg, 10 * deg, 130 * deg)
    target = Kep(7_000_000, 0.00149, 90 * deg, 12 * deg, 130 * deg)

    mans = noncoplanar_analytical(initial, target, mu)

    angle1 = 1.570796326794897
    impulse1T = 0.0
    impulse1R = -2.9806909493887241
    impulse1N = -131.69678458903252

    angle2 = 4.71238898038469
    impulse2T = 0.0
    impulse2R = 2.9806909493887241
    impulse2N = 131.69678458903252

    ref = reference_orbit(initial, target, mu)
    dev = trans_devs(initial, target)
    deltaV = float(np.sqrt(sqr(dev.i) + sqr(dev.ex) / 4.0 + sqr(dev.ey)))

    assert len(mans) == 2
    assert np.isclose(mans[0].angle, angle1, atol=tolerance)
    assert_vec_close(mans[0].dv, impulse1R, impulse1T, impulse1N, atol=100 * tolerance)

    assert np.isclose(mans[1].angle, angle2, atol=tolerance)
    assert_vec_close(mans[1].dv, impulse2R, impulse2T, impulse2N, atol=100 * tolerance)

    assert np.isclose(
        deltaV * ref.v,
        np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[1].dv),
        atol=tolerance,
    )


def test_nodal_case_with_ex():
    # Узловой случай с НЕнулевой проекцией вектора эксцентриситета на ось Ox
    initial = Kep(7_000_000, 0.00228, 20 * deg, 10 * deg, 130 * deg)
    target = Kep(7_010_000, 0.00149, 150 * deg, 15 * deg, 130 * deg)

    mans = noncoplanar_analytical(initial, target, mu)

    angle1 = 2.792526803190928
    impulse1T = 9.1659903468321371
    impulse1R = -0.18586788072459307
    impulse1N = -465.86566355536723

    angle2 = 5.934119456780721
    impulse2T = -3.781722352266204
    impulse2R = 0.076685736347897451
    impulse2N = 192.20777312181514

    ref = reference_orbit(initial, target, mu)
    dev = trans_devs(initial, target)
    deltaV = float(np.sqrt(sqr(dev.i) + sqr(dev.ex) / 4.0 + sqr(dev.ey)))

    assert len(mans) == 2
    assert np.isclose(mans[0].angle, angle1, atol=tolerance)
    assert_vec_close(mans[0].dv, impulse1R, impulse1T, impulse1N, atol=100 * tolerance)

    assert np.isclose(mans[1].angle, angle2, atol=tolerance)
    assert_vec_close(mans[1].dv, impulse2R, impulse2T, impulse2N, atol=100 * tolerance)

    assert np.isclose(
        deltaV * ref.v,
        np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[1].dv),
        atol=tolerance,
    )


def test_non_degenerate_case_without_e():
    # Невырожденный случай с нулевым вектором эксцентриситета
    initial = Kep(7_000_000, 0.0, 0 * deg, 10 * deg, 130 * deg)
    target = Kep(8_000_000, 0.0, 0 * deg, 15 * deg, 130 * deg)

    mans = noncoplanar_analytical(initial, target, mu)

    angle1 = 5.551115123125784e-17
    impulse1T = 243.00599443909471
    impulse1R = 0.0
    impulse1N = 317.99317810613087

    angle2 = 3.141592653589793
    impulse2T = 243.00599443909471
    impulse2R = 0.0
    impulse2N = -317.99317810613087

    ref = reference_orbit(initial, target, mu)
    dev = trans_devs(initial, target)
    multiplier1 = sqr(dev.i) - sqr(dev.ex) - sqr(dev.ey) + sqr(dev.a)
    denominator = np.sqrt(sqr(multiplier1) + 4.0 * sqr(dev.i) * sqr(dev.ey))
    deltaV = float(
        np.sqrt((sqr(dev.i) + sqr(dev.ex) + sqr(dev.ey) - sqr(dev.a) / 2.0 + denominator) / 2.0)
    )

    assert len(mans) == 2
    assert np.isclose(mans[0].angle, angle1, atol=tolerance)
    assert_vec_close(mans[0].dv, impulse1R, impulse1T, impulse1N, atol=100 * tolerance)

    assert np.isclose(mans[1].angle, angle2, atol=tolerance)
    assert_vec_close(mans[1].dv, impulse2R, impulse2T, impulse2N, atol=100 * tolerance)

    assert np.isclose(
        deltaV * ref.v,
        np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[1].dv),
        atol=tolerance * 10,
    )


def test_non_degenerate_case_without_ey():
    # Невырожденный случай с нулевой проекцией эксцентриситета на Oy
    initial = Kep(7_000_000, 0.0, 0 * deg, 10 * deg, 130 * deg)
    target = Kep(8_000_000, 0.05, 0 * deg, 15 * deg, 130 * deg)

    mans = noncoplanar_analytical(initial, target, mu)

    angle1 = 5.5511151231257839e-17
    impulse1R = 0.0
    impulse1T = 334.13324235375524
    impulse1N = 317.99317810613087

    angle2 = 3.1415926535897931
    impulse2R = 0.0
    impulse2T = 151.87874652443418
    impulse2N = -317.99317810613087

    ref = reference_orbit(initial, target, mu)
    dev = trans_devs(initial, target)
    multiplier1 = sqr(dev.i) - sqr(dev.ex) - sqr(dev.ey) + sqr(dev.a)
    denominator = np.sqrt(sqr(multiplier1) + 4.0 * sqr(dev.i) * sqr(dev.ey))
    deltaV = float(
        np.sqrt((sqr(dev.i) + sqr(dev.ex) + sqr(dev.ey) - sqr(dev.a) / 2.0 + denominator) / 2.0)
    )

    assert len(mans) == 2
    assert np.isclose(mans[0].angle, angle1, atol=tolerance)
    assert_vec_close(mans[0].dv, impulse1R, impulse1T, impulse1N, atol=100 * tolerance)

    assert np.isclose(mans[1].angle, angle2, atol=tolerance)
    assert_vec_close(mans[1].dv, impulse2R, impulse2T, impulse2N, atol=100 * tolerance)

    dV = np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[1].dv)
    # относительная погрешность 2%
    assert np.isclose(deltaV * ref.v, dV, rtol=0.02)


def test_non_degenerate_case_with_e():
    # Невырожденный случай с НЕнулевым вектором эксцентриситета
    initial = Kep(7_000_000, 0.00228, 20 * deg, 10 * deg, 130 * deg)
    target = Kep(8_000_000, 0.00149, 150 * deg, 15 * deg, 130 * deg)

    mans = noncoplanar_analytical(initial, target, mu)

    angle1 = 2.792704971514334
    impulse1T = 249.2625670546304
    impulse1R = -0.03902898697930736
    impulse1N = -326.1804194917825

    angle2 = 5.933931871555196
    impulse2T = 236.74942182355903
    impulse2R = 0.037069706097045442
    impulse2N = 309.8059473483662

    ref = reference_orbit(initial, target, mu)
    dev = trans_devs(initial, target)
    multiplier1 = sqr(dev.i) - sqr(dev.ex) - sqr(dev.ey) + sqr(dev.a)
    denominator = np.sqrt(sqr(multiplier1) + 4.0 * sqr(dev.i) * sqr(dev.ey))
    deltaV = float(
        np.sqrt((sqr(dev.i) + sqr(dev.ex) + sqr(dev.ey) - sqr(dev.a) / 2.0 + denominator) / 2.0)
    )

    assert len(mans) == 2
    assert np.isclose(mans[0].angle, angle1, atol=tolerance)
    assert_vec_close(mans[0].dv, impulse1R, impulse1T, impulse1N, atol=100 * tolerance)

    assert np.isclose(mans[1].angle, angle2, atol=tolerance)
    assert_vec_close(mans[1].dv, impulse2R, impulse2T, impulse2N, atol=100 * tolerance)

    assert np.isclose(
        deltaV * ref.v,
        np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[1].dv),
        atol=tolerance * 10,
    )


def test_singular_case():
    # Особый случай
    initial = Kep(7_000_000, 0.00228, 90 * deg, 10 * deg, 130 * deg)
    target = Kep(7_010_000, 0.09149, 90 * deg, 12 * deg, 130 * deg)

    mans = noncoplanar_analytical(initial, target, mu)

    ref = reference_orbit(initial, target, mu)
    dev = trans_devs(initial, target)
    deltaV = float(np.sqrt(sqr(np.sqrt(3.0) * dev.i + dev.ey) + sqr(dev.ex)) / 2.0)
    dV = np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[0].dv) + np.linalg.norm(mans[2].dv)
    assert np.isclose(deltaV * ref.v, dV, rtol=0.07)
