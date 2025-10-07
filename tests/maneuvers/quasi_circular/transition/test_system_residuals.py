from __future__ import annotations

import math

import numpy as np

from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.maneuvers.quasi_circular.reference_orbit import TransDevs, trans_devs
from maneuvering.maneuvers.quasi_circular.transition.coplanar.coplanar import solve_coplanar_sys
from maneuvering.maneuvers.quasi_circular.transition.noncoplanar.noncoplanar import (
    solve_noncoplanar_sys,
)
from maneuvering.orbit.keplerian import Kep

deg: float = math.pi / 180.0
tol: float = float(np.finfo(float).eps * 1e2)


def calc_left_hand_side(mans: list[Maneuver]) -> np.ndarray:
    lhs = np.zeros(5, dtype=float)
    for m in mans:
        s, c = math.sin(m.angle), math.cos(m.angle)
        dvx, dvy, dvz = float(m.dv[0]), float(m.dv[1]), float(m.dv[2])
        lhs[0] += dvx * s + 2.0 * dvy * c
        lhs[1] += -dvx * c + 2.0 * dvy * s
        lhs[2] += 2.0 * dvy
        lhs[3] += -dvz * s
        lhs[4] += dvz * c
    return lhs


def check_non_copl_sys_residuals_transdevs(devs: TransDevs, tol: float) -> None:
    mans = solve_noncoplanar_sys(devs)
    lhs = calc_left_hand_side(mans)
    assert math.isfinite(lhs[0])
    assert math.isfinite(lhs[1])
    assert math.isfinite(lhs[2])
    assert math.isfinite(lhs[3])
    assert math.isfinite(lhs[4])
    assert abs(lhs[0] - devs.ex) <= tol
    assert abs(lhs[1] - devs.ey) <= tol
    assert abs(lhs[2] - devs.a) <= tol
    assert abs(lhs[3] - 0.0) <= tol
    assert abs(lhs[4] - devs.i) <= tol


def check_non_copl_sys_residuals_orbits(oi: Kep, ot: Kep, tol: float) -> None:
    devs = trans_devs(oi, ot)
    check_non_copl_sys_residuals_transdevs(devs, tol)


def check_copl_sys_residuals_transdevs(devs: TransDevs, tol: float) -> None:
    mans = solve_coplanar_sys(devs)
    lhs = calc_left_hand_side(mans)
    assert math.isfinite(lhs[0])
    assert math.isfinite(lhs[1])
    assert math.isfinite(lhs[2])
    assert abs(lhs[0] - devs.ex) <= tol
    assert abs(lhs[1] - devs.ey) <= tol
    assert abs(lhs[2] - devs.a) <= tol


def check_copl_sys_residuals_orbits(oi: Kep, ot: Kep, tol: float) -> None:
    devs = trans_devs(oi, ot)
    check_copl_sys_residuals_transdevs(devs, tol)


def test_system_residuals_cases() -> None:
    # Компланарный переход, непересекающиеся орбиты
    oi = Kep(a=6_566_000.0, e=0.00228, i=0.0, w=20 * deg, raan=130 * deg)
    ot = Kep(a=6_721_000.0, e=0.00149, i=0.0, w=150 * deg, raan=130 * deg)
    check_copl_sys_residuals_orbits(oi, ot, tol)

    # Компланарный переход, пересекающиеся орбиты
    oi = Kep(a=6_566_000.0, e=0.00228, i=0.0, w=20 * deg, raan=130 * deg)
    ot = Kep(a=6_576_000.0, e=0.00149, i=0.0, w=150 * deg, raan=130 * deg)
    check_copl_sys_residuals_orbits(oi, ot, tol)

    # Некомпланарный, узловой, ΔE_x≈0
    oi = Kep(a=7_000_000.0, e=0.00228, i=10 * deg, w=90 * deg, raan=130 * deg)
    ot = Kep(a=7_000_000.0, e=0.00149, i=12 * deg, w=90 * deg, raan=130 * deg)
    check_non_copl_sys_residuals_orbits(oi, ot, tol)

    # Некомпланарный, узловой, ΔE_x≠0
    oi = Kep(a=7_000_000.0, e=0.00228, i=10 * deg, w=20 * deg, raan=130 * deg)
    ot = Kep(a=7_010_000.0, e=0.00149, i=15 * deg, w=150 * deg, raan=130 * deg)
    check_non_copl_sys_residuals_orbits(oi, ot, tol)

    # Некомпланарный, невырожденный, ΔE=0
    oi = Kep(a=7_000_000.0, e=0.0, i=10 * deg, w=0.0, raan=130 * deg)
    ot = Kep(a=8_000_000.0, e=0.0, i=15 * deg, w=0.0, raan=130 * deg)
    check_non_copl_sys_residuals_orbits(oi, ot, tol)

    # Некомпланарный, невырожденный, ΔE_y≈0
    oi = Kep(a=7_000_000.0, e=0.0, i=10 * deg, w=0.0, raan=130 * deg)
    ot = Kep(a=8_000_000.0, e=0.05, i=15 * deg, w=0.0, raan=130 * deg)
    check_non_copl_sys_residuals_orbits(oi, ot, tol)

    # Некомпланарный, невырожденный, ΔE≠0
    oi = Kep(a=7_000_000.0, e=0.00228, i=10 * deg, w=20 * deg, raan=130 * deg)
    ot = Kep(a=8_000_000.0, e=0.00149, i=15 * deg, w=150 * deg, raan=130 * deg)
    check_non_copl_sys_residuals_orbits(oi, ot, tol * 100)

    # Некомпланарный, особый случай
    oi = Kep(a=7_000_000.0, e=0.00228, i=10 * deg, w=90 * deg, raan=130 * deg)
    ot = Kep(a=7_010_000.0, e=0.09149, i=12 * deg, w=90 * deg, raan=130 * deg)
    check_non_copl_sys_residuals_orbits(oi, ot, tol)


def test_system_residuals_grid() -> None:
    # Диапазоны как в C++ (включительно, шаги подобраны для совпадения сетки)
    dEx_vals = np.round(np.arange(-0.07, 0.0700001, 0.005), 10)
    dEy_vals = np.round(np.arange(-0.07, 0.0700001, 0.005), 10)
    dA_vals = np.round(np.arange(-1.0, 1.000001, 0.05), 10)
    dI_vals = np.round(np.arange(-5 * deg, 5 * deg + 1e-12, 0.1 * deg), 12)

    for dEx in dEx_vals:
        for dEy in dEy_vals:
            for dA in dA_vals:
                for dI in dI_vals:
                    dE = math.hypot(dEx, dEy)
                    devs = TransDevs(
                        ex=float(dEx), ey=float(dEy), e=float(dE), a=float(dA), i=float(abs(dI))
                    )
                    if abs(dI) < tol:
                        check_copl_sys_residuals_transdevs(devs, tol * 100.0)
                    else:
                        check_non_copl_sys_residuals_transdevs(devs, 1e-7)
