# src/maneuvering/maneuvers/quasi_circular/noncoplanar.py
from __future__ import annotations

import math

import numpy as np

from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.maneuvers.quasi_circular.reference_orbit import (
    TransDevs,
    reference_orbit,
    trans_devs,
)
from maneuvering.maneuvers.quasi_circular.transition.noncoplanar.intersection_angle import (
    intersection_angle,
)
from maneuvering.orbit.keplerian import Kep
from maneuvering.types import Scalar
from maneuvering.utils.math_tools import normalize_angle

# Почему: единообразие с С++ (численная толерантность ~ 100*eps double)
_MACHINE_TOL: float = np.finfo(float).eps * 100.0


def _sqr(x: Scalar) -> Scalar:
    return x * x


# ============================== Узловой случай ==============================


def nodal_case_without_ex(devs: TransDevs) -> list[Maneuver]:
    """
    Узловой случай при нулевой проекции вектора эксцентриситета на Ox (ΔE_x ≈ 0).

    Формулы
    -------
    dv₁ = [ -ΔE_y/2, 0, +Δi/2 ], angle₁ = 0
    dv₂ = [ +ΔE_y/2, 0, -Δi/2 ], angle₂ = π

    Возвращает
    -------
    list[Maneuver] : два манёвра (безразмерные dv, углы в рад).
    """
    imp1 = np.array([-devs.ey / 2.0, 0.0, devs.i / 2.0])
    man1 = Maneuver(dv=imp1, angle=0.0)

    imp2 = np.array([devs.ey / 2.0, 0.0, -devs.i / 2.0])
    man2 = Maneuver(dv=imp2, angle=math.pi)

    return [man1, man2]


def nodal_case_with_ex(devs: TransDevs) -> list[Maneuver]:
    """
    Узловой случай при ненулевой проекции эксцентриситета на Ox (ΔE_x ≠ 0).

    Формулы
    -------
    imp1_t = (ΔA + ΔE_x)/4
    imp1_r = -(ΔA + ΔE_x)·ΔE_y / (2·ΔE_x)
    imp1_n = +(ΔA + ΔE_x)·Δi   / (2·ΔE_x)
    angle₁ = 0

    imp2_t = (ΔA − ΔE_x)/4
    imp2_r = -(ΔA − ΔE_x)·ΔE_y / (2·ΔE_x)
    imp2_n = +(ΔA − ΔE_x)·Δi   / (2·ΔE_x)
    angle₂ = π
    """
    ex = devs.ex
    t1 = (devs.a + ex) / 4.0
    r1 = -(devs.a + ex) * devs.ey / (2.0 * ex)
    n1 = (devs.a + ex) * devs.i / (2.0 * ex)
    man1 = Maneuver(dv=np.array([r1, t1, n1]), angle=0.0)

    t2 = (devs.a - ex) / 4.0
    r2 = -(devs.a - ex) * devs.ey / (2.0 * ex)
    n2 = (devs.a - ex) * devs.i / (2.0 * ex)
    man2 = Maneuver(dv=np.array([r2, t2, n2]), angle=math.pi)

    return [man1, man2]


def noncoplanar_nodal(devs: TransDevs, tol: Scalar = _MACHINE_TOL) -> list[Maneuver]:
    """
    Узловой тип задачи (node-to-node). Выбор подслучая по ΔE_x.

    Критерий
    --------
    |ΔE_x| < tol → nodal_case_without_ex
    иначе → nodal_case_with_ex
    """
    return nodal_case_without_ex(devs) if abs(devs.ex) < tol else nodal_case_with_ex(devs)


# ====================== Невырожденный / Вырожденный / Особый ======================


def nondegenerate_case_without_e(devs: TransDevs) -> list[Maneuver]:
    """
    Невырожденный случай при нулевом векторе эксцентриситета (ΔE=0).

    Формулы
    -------
    dv₁ = [0, ΔA/4, +Δi/2], angle₁ = 0
    dv₂ = [0, ΔA/4, −Δi/2], angle₂ = π
    """
    imp1 = np.array([0.0, devs.a / 4.0, devs.i / 2.0])
    man1 = Maneuver(dv=imp1, angle=0.0)

    imp2 = np.array([0.0, devs.a / 4.0, -devs.i / 2.0])
    man2 = Maneuver(dv=imp2, angle=math.pi)

    return [man1, man2]


def nondegenerate_case_without_ey(devs: TransDevs) -> list[Maneuver]:
    """
    Невырожденный случай при нулевой проекции эксцентриситета на Oy (ΔE_y≈0).

    Формулы
    -------
    dv₁ = [0, (ΔA+ΔE_x)/4, +Δi/2], angle₁ = 0
    dv₂ = [0, (ΔA−ΔE_x)/4, −Δi/2], angle₂ = π
    """
    imp1 = np.array([0.0, (devs.a + devs.ex) / 4.0, devs.i / 2.0])
    man1 = Maneuver(dv=imp1, angle=0.0)

    imp2 = np.array([0.0, (devs.a - devs.ex) / 4.0, -devs.i / 2.0])
    man2 = Maneuver(dv=imp2, angle=math.pi)

    return [man1, man2]


def nondegenerate_case_with_e(devs: TransDevs) -> list[Maneuver]:
    """
    Общий невырожденный случай (ΔE_x≠0, ΔE_y≠0) из Барнова/оптимального управления.

    Возвращает
    -------
    list[Maneuver] : два манёвра (безразмерные dv, углы).
    """
    e2 = _sqr(devs.e)
    a2 = _sqr(devs.a)
    i2 = _sqr(devs.i)
    ex2 = _sqr(devs.ex)
    ey2 = _sqr(devs.ey)

    mult1 = i2 - e2 + a2
    denom = math.sqrt(_sqr(mult1) + 4.0 * i2 * ey2)
    delta_v = math.sqrt((i2 + e2 - a2 / 2.0 + denom) / 2.0)

    # λ-параметры
    lam1 = devs.a / (2.0 * delta_v) * (-0.5 + mult1 / denom)
    lam2 = devs.ey / (2.0 * delta_v) * (1.0 - (-i2 - e2 + a2) / denom)
    lam3 = devs.ex / (2.0 * delta_v) * (1.0 - mult1 / denom)
    lam4 = -devs.ey / (2.0 * delta_v) * (2.0 * devs.i * devs.ex / denom)
    lam5 = devs.i / (2.0 * delta_v) * (1.0 + (i2 - ex2 + ey2 + a2) / denom)

    theta0 = math.atan2(lam2, lam3)
    m2 = math.sqrt(_sqr(lam2) + _sqr(lam3))
    num = 4.0 * lam1 * (m2**3)
    den = _sqr(lam3 * lam4 - lam2 * lam5) - 3.0 * (m2**4)
    # Почему: числ. защита acos
    cos_dtheta = np.clip(num / den, -1.0, 1.0)
    dtheta = math.acos(cos_dtheta)

    theta1 = theta0 + dtheta
    theta2 = 2.0 * theta0 - theta1

    # Направляющие косинусы
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    dir_r = -lam2 * c1 + lam3 * s1
    dir_t = 2.0 * lam1 + 2.0 * lam2 * s1 + 2.0 * lam3 * c1
    dir_n = lam4 * s1 + lam5 * c1

    # Деление общего ΔV
    s1_abs = abs(math.sin(theta1))
    s2_abs = abs(math.sin(theta2))
    delta = 0.5 if (s1_abs == 0.0 and s2_abs == 0.0) else (s1_abs / (s1_abs + s2_abs))

    dv1 = delta_v * (1.0 - delta)
    dv2 = delta_v * delta

    imp1 = np.array([dir_r, dir_t, dir_n]) * dv1
    man1 = Maneuver(dv=imp1, angle=float(theta1))

    imp2 = np.array([-dir_r, dir_t, -dir_n]) * dv2
    man2 = Maneuver(dv=imp2, angle=float(theta2))

    return [man1, man2]


def noncoplanar_degenerate(devs: TransDevs, tol: Scalar = _MACHINE_TOL) -> list[Maneuver]:
    """
    Вырожденный случай: ветвление по ΔE_y≈0, ΔE_x≈0 с редкими частными случаями.
    """
    ey_zero = abs(devs.ey) < tol
    ex_zero = abs(devs.ex) < tol
    if ey_zero and ex_zero:
        return nondegenerate_case_without_e(devs)
    if ey_zero:
        return nondegenerate_case_without_ey(devs)
    return nondegenerate_case_with_e(devs)


def noncoplanar_singular(devs: TransDevs) -> list[Maneuver]:
    """
    Особый (singular) случай: 3 импульса. Решение 3×4 СЛАУ через МНК.

    Формулы
    -------
    θy = √3·Δi + ΔE_y
    θx = ΔE_x
    θ0 = atan2(θy, θx)
    φ₁=θ0, φ₂=θ0+π, φ₃=0
    Решаем A·x=b:
      A = [[2c1, 2c2, 2,  0],
           [2s1, 2s2, 0, -1],
           [  2,   2, 2,  0]],  b=[ΔE_x, ΔE_y, ΔA]
    dv: imp1=[0, x0, 0]@φ₁; imp2=[0, x1, 0]@φ₂; imp3=[x3, x2, Δi]@φ₃
    """
    theta_y = math.sqrt(3.0) * devs.i + devs.ey
    theta_x = devs.ex
    theta0 = math.atan2(theta_y, theta_x)

    phi1 = theta0
    phi2 = theta0 + math.pi
    phi3 = 0.0

    c1, s1 = math.cos(phi1), math.sin(phi1)
    c2, s2 = math.cos(phi2), math.sin(phi2)

    A = np.array(
        [
            [2.0 * c1, 2.0 * c2, 2.0, 0.0],
            [2.0 * s1, 2.0 * s2, 0.0, -1.0],
            [2.0, 2.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    b = np.array([devs.ex, devs.ey, devs.a], dtype=float)

    # МНК-решение (минимальная норма)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    imp1 = np.array([0.0, sol[0], 0.0])
    imp2 = np.array([0.0, sol[1], 0.0])
    imp3 = np.array([sol[3], sol[2], devs.i])

    mans = [
        Maneuver(dv=imp1, angle=float(phi1)),
        Maneuver(dv=imp2, angle=float(phi2)),
        Maneuver(dv=imp3, angle=float(phi3)),
    ]
    return mans


def solve_noncoplanar_sys(devs: TransDevs) -> list[Maneuver]:
    """
    Классификация задачи (около круговая аппроксимация), см. Барнов.

    Условия
    -------
    cond1: 3·ΔE_y² ≤ Δi²
    cond2: ΔA² ≤ ΔE_x²
    cond3: ΔE² + (2/√3)·ΔE_y·Δi − Δi² ≤ ΔA²

    Возвращает
    -------
    list[Maneuver] : узловой / вырожденный / особый.
    """
    cond1 = 3.0 * _sqr(devs.ey) <= _sqr(devs.i)
    cond2 = _sqr(devs.a) <= _sqr(devs.ex)
    cond3 = _sqr(devs.e) + (2.0 / math.sqrt(3.0)) * devs.ey * devs.i - _sqr(devs.i) <= _sqr(devs.a)

    if cond1 and cond2:
        return noncoplanar_nodal(devs)
    if (not cond2) and cond3:
        return noncoplanar_degenerate(devs)
    return noncoplanar_singular(devs)


def noncoplanar_analytical(oi: Kep, ot: Kep, mu: Scalar) -> list[Maneuver]:
    """
    Аналитическое решение некомпланарного перехода (около круговое приближение).

    Алгоритм
    --------
    1) devs = trans_devs(oi, ot)  — отклонения от опорной круговой орбиты.
    2) mans = solve_noncoplanar_sys(devs) — dv безразмерные.
    3) intersect_angle = planes_intersection_anomaly(oi, ot) (fallback: 2π − ω_i).
    4) v = sqrt(μ / a_ref) из reference_orbit(oi, ot, mu).
    5) Масштабировать dv := dv·v; angle := normalize(intersect_angle + angle); отсортировать.
    """
    devs = trans_devs(oi, ot)
    mans = solve_noncoplanar_sys(devs)

    intersect_angle = intersection_angle(oi, ot)
    ref = reference_orbit(oi, ot, mu)

    scaled_sorted = [
        Maneuver(dv=m.dv * ref.v, angle=normalize_angle(intersect_angle + m.angle)) for m in mans
    ]
    scaled_sorted.sort(key=lambda m: m.angle)
    return scaled_sorted
