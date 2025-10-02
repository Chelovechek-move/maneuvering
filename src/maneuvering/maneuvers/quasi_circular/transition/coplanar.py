from __future__ import annotations

import numpy as np

from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.maneuvers.quasi_circular.reference_orbit import (
    TransDevs,
    reference_orbit,
    trans_devs,
)
from maneuvering.orbit.keplerian import Kep
from maneuvering.types import Scalar
from maneuvering.utils.math_tools import normalize_angle


def coplanar_intersected(devs: TransDevs) -> list[Maneuver]:
    """
    Рассчитывает манёвры для случая пересекающихся орбит (ΔE > |ΔA|).

    Формулы
    -------
    angle₁ = atan2(ΔE_y, ΔE_x)
    angle₂ = angle₁ + π
    dv₁ = [(0), (ΔA + ΔE)/4, (0)]     (вдоль t)
    dv₂ = [(0), (ΔA − ΔE)/4, (0)]     (вдоль t)

    Здесь dv заданы в безразмерной форме (относительно круговой скорости опорной орбиты).
    Масштабирование в м/с выполняется в `coplanar_analytical()`.

    Параметры
    ----------
    devs : TransDevs
        Отклонения от опорной круговой орбиты:
        ΔE_x, ΔE_y, ΔE = sqrt(ΔE_x² + ΔE_y²), ΔA = (a_t − a_i)/a_ref, Δi [рад].

    Возвращает
    -------
    list[Maneuver]
        Два манёвра с безразмерными dv и углами точки приложения [рад].
    """
    ang1 = # СЮДА НАДО НАПИСАТЬ КОД ...
    imp1 = # СЮДА НАДО НАПИСАТЬ КОД ...
    man1 = Maneuver(dv=imp1, angle=ang1)

    ang2 = # СЮДА НАДО НАПИСАТЬ КОД ...
    imp2 = # СЮДА НАДО НАПИСАТЬ КОД ...
    man2 = Maneuver(dv=imp2, angle=ang2)

    return [man1, man2]


def coplanar_non_intersecting(devs: TransDevs, ang1: Scalar | None = None) -> list[Maneuver]:
    """
    Рассчитывает манёвры для случая непересекающихся орбит (ΔE ≤ |ΔA|).

    Формулы
    -------
    angle₁ = atan2(ΔE_y, ΔE_x)

    dvt₁ = ((ΔE² − ΔA²) / (ΔE_y·sin(angle₁) + ΔE_x·cos(angle₁) − ΔA)) / 4
    dv₁  = [(0), dvt₁, (0)]

    dvt₂ = ΔA/2 − dvt₁
    dv₂  = [(0), dvt₂, (0)]

    angle₂ = atan2( (ΔE_y/2 − dvt₁·sin(angle₁)) / dvt₂,
                    (ΔE_x/2 − dvt₁·cos(angle₁)) / dvt₂ )

    Здесь dv заданы в безразмерной форме (относительно круговой скорости опорной орбиты).
    Масштабирование в м/с выполняется в `coplanar_analytical()`.

    Параметры
    ----------
    devs : TransDevs
        Отклонения от опорной круговой орбиты:
        ΔE_x, ΔE_y, ΔE = sqrt(ΔE_x² + ΔE_y²), ΔA = (a_t − a_i)/a_ref, Δi [рад].

    Возвращает
    -------
    list[Maneuver]
        Два манёвра с безразмерными dv и углами точки приложения [рад].
    """
    ang1 = float(np.arctan2(devs.ey, devs.ex)) if ang1 is None else normalize_angle(ang1)

    # СЮДА НАДО НАПИСАТЬ КОД ...
    imp1 = # СЮДА НАДО НАПИСАТЬ КОД ...
    man1 = Maneuver(dv=imp1, angle=ang1)

    dvt2 = # СЮДА НАДО НАПИСАТЬ КОД ...
    imp2 = np.array([0.0, dvt2, 0.0])

    # СЮДА НАДО НАПИСАТЬ КОД ...
    ang2 = # СЮДА НАДО НАПИСАТЬ КОД ...
    man2 = Maneuver(dv=imp2, angle=ang2)

    return [man1, man2]


def solve_coplanar_sys(devs: TransDevs, tol: Scalar = 2.220446049250313e-14) -> list[Maneuver]:
    """
    Выбирает алгоритм решения (пустой/пересекающиеся/непересекающиеся) по параметрам отклонений.

    Критерии
    --------
    isCoincide: |ΔE| < tol и |ΔA| < tol → манёвры не нужны → []
    isIntersect: ΔE > |ΔA| → пересекающиеся  → `coplanar_intersected`
    иначе непересекающиеся → `coplanar_non_intersecting`

    Параметры
    ----------
    devs : TransDevs
        Отклонения от опорной круговой орбиты.
    tol : Scalar, optional
        Численная точность для определения «совпадающих» орбит, [1].
        По умолчанию ~ `machine epsilon * 100`.

    Возвращает
    -------
    list[Maneuver]
        Список манёвров с безразмерными dv и углами точки приложения [рад].
    """
    is_coincide = (abs(devs.e) < tol) and (abs(devs.a) < tol)
    is_intersect = devs.e > abs(devs.a)
    if is_coincide:
        return []
    return coplanar_intersected(devs) if is_intersect else coplanar_non_intersecting(devs)


def coplanar_analytical(oi: Kep, ot: Kep, mu: Scalar) -> list[Maneuver]:
    """
    Рассчитывает аналитическое решение задачи компланарного перехода.

    Алгоритм
    --------
    1) Вычислить отклонения от опорной круговой орбиты:
         devs = trans_devs(oi, ot)
       где:
         ΔE_x = e_t·cos(ω_t) − e_i·cos(ω_i),
         ΔE_y = e_t·sin(ω_t) − e_i·sin(ω_i),
         ΔE = sqrt(ΔE_x² + ΔE_y²),
         a_ref = (a_i + a_t)/2,
         ΔA = (a_t − a_i)/a_ref.
    2) Решить систему (пересечение/непересечение): mans = solve_coplanar_sys(devs).
       На этом шаге dv безразмерные (относительно круговой скорости).
    3) Угол точки приложения сдвигается на угол пересечения:
         intersectAngle = normalize(2π − ω_i).
    4) Круговая скорость опорной орбиты:
         v = sqrt(μ / a_ref), где a_ref = (a_i + a_t)/2.
    5) Масштабировать импульсы и нормализовать углы:
         dv := dv * v,  angle := normalize(intersectAngle + angle).
       Отсортировать по углу по возрастанию.

    Параметры
    ----------
    oi : Kep
        Начальная орбита {a [м], e [1], w [рад], i [рад], raan [рад]}.
    ot : Kep
        Целевая орбита {a [м], e [1], w [рад], i [рад], raan [рад]}.
    mu : Scalar
        Гравитационный параметр центрального тела μ, [м³/с²].

    Возвращает
    -------
    list[Maneuver]
        Список из 0 или 2 манёвров:
        - dv : Vector3 — импульс в орбитальной СК {r, t, n}, [м/с]
        - angle : Scalar — истинная широта точки приложения (рад), нормализована в [0, 2π).
    """
    devs = trans_devs(oi, ot)
    mans = solve_coplanar_sys(devs)

    intersect_angle = normalize_angle(2.0 * np.pi - oi.w)

    ref = reference_orbit(oi, ot, mu)
    scaled_sorted = [
        Maneuver(dv=m.dv * ref.v, angle=normalize_angle(intersect_angle + m.angle)) for m in mans
    ]
    scaled_sorted.sort(key=lambda m: m.angle)
    return scaled_sorted
