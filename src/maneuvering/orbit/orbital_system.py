from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maneuvering.types import ArrayF64, Vector3
from maneuvering.utils.math_tools import normalize


@dataclass(frozen=True, slots=True)
class OrbitalSystem:
    """
    Ортонормированный базис орбитальной системы координат {r, t, n}.

    Атрибуты
    --------
    r : Vector3
        Единичный вектор, сонаправленный радиус-вектору (в координатах ECI).
    t : Vector3
        Единичный вектор, дополняющий до правой тройки: `t = n × r`.
    n : Vector3
        Единичный вектор, сонаправленный орбитальному моменту: `n ∝ r × v`.
    """
    r: Vector3
    t: Vector3
    n: Vector3


def calc_orb_sys(pos: Vector3, vel: Vector3) -> OrbitalSystem:
    """
    Строит базис орбитальной СК по вектору положения и скорости в ECI.

    Parameters
    ----------
    position : Vector3
        Радиус-вектор в ECI, м.
    velocity : Vector3
        Скорость в ECI, м/с.

    Returns
    -------
    OrbitalSystem
        Ортонормированный базис {r, t, n} в координатах ECI.

    Raises
    ------
    ValueError
        Если `position` нулевой или `position × velocity` вырожден.

    Notes
    -----
    r = position / ||position||, n = (position × velocity)/||...||, t = n × r.
    """
    r = normalize(pos)
    h = np.cross(pos, vel)
    n = normalize(h)
    t = np.cross(n, r)
    return OrbitalSystem(r=r, t=t, n=n)


def rot_mat_orb_to_eci(o: OrbitalSystem) -> ArrayF64:
    """
    Рассчитывает матрицу перехода (поворота) из орбитальной СК в ECI.

    Столбцы матрицы — базисные векторы {r, t, n}, заданные в координатах ECI.

    Parameters
    ----------
    o : OrbitalSystem
        Базис в орбитальной СК.

    Returns
    -------
    ArrayF64
        Матрица формы (3, 3), ортогональная.
    """
    R = np.column_stack((o.r, o.t, o.n))
    return R


def orb_to_eci(vec_orb: Vector3, pos: Vector3, vel: Vector3) -> Vector3:
    """
    Перевести вектор из орбитальной СК в ECI.

    Parameters
    ----------
    vec_orb : Vector3
        Координаты вектора в орбитальной СК.
    pos, vel : Vector3
        Положение и скорость тела в ECI.

    Returns
    -------
    Vector3
        Вектор в ECI.
    """
    o = calc_orb_sys(pos, vel)
    R = rot_mat_orb_to_eci(o)
    return R @ vec_orb


def eci_to_orb(vec_eci: Vector3, pos: Vector3, vel: Vector3) -> Vector3:
    """
    Перевести вектор из ECI в орбитальную СК.

    Parameters
    ----------
    vec_eci : Vector3
        Вектор в ECI.
    pos, vel : Vector3
        Положение и скорость тела в ECI.

    Returns
    -------
    Vector3
        Координаты в орбитальной СК.
    """
    o = calc_orb_sys(pos, vel)
    R = rot_mat_orb_to_eci(o)
    # Обратное преобразование для ортогональной R — это R^T
    return R.T @ vec_eci
