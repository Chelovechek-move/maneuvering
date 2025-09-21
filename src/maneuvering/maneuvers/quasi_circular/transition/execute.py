from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from maneuvering.types import Scalar, Vector3
from maneuvering.utils.math_tools import normalize_angle
from maneuvering.orbit.keplerian import KepTrue
from maneuvering.orbit.convert_kep_cart import convert_kep_true_to_cart
from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.maneuvers.apply_impulse import apply_impulse_orb


def execute(oi: KepTrue, maneuvers: List[Maneuver], mu: Scalar) -> KepTrue:
    """
    Применяет последовательность манёвров и возвращает финальную орбиту.

    Пассивное движение аппарата прогнозируется в поле точечного потенциала

    Parameters
    ----------
    oi : KepTrue
        Начальные истинные кеплеровы элементы.
    maneuvers : list[Maneuver]
        Список манёвров. Для каждого m:
          - m.angle — истинная широта u = w + nu, [рад];
          - m.dv    — импульс Δv в орбитальной СК {r, t, n}, [м/с].
    mu : Scalar
        Гравитационный параметр центрального тела, [м³/с²].

    Returns
    -------
    KepTrue
        Финальные истинные элементы после применения всех манёвров.

    Warning
    -------
    - Необходимо, чтобы истинная широта приложения первого манёвра в maneuvers была больше или равна истинной широте
    точки oi.
    - Необходимо, чтобы манёвры в maneuvers были отсортированы в порядке возрастания аргумента широты.
    """
    cur = KepTrue(a=oi.a, e=oi.e, w=oi.w, i=oi.i, raan=oi.raan, nu=oi.nu)
    u_cur = normalize_angle(cur.w + cur.nu)

    for m in maneuvers:
        du = (m.angle - u_cur) % (2.0 * math.pi)
        cur = KepTrue(a=cur.a, e=cur.e, w=cur.w, i=cur.i, raan=cur.raan, nu=normalize_angle(cur.nu + du))
        cur = apply_impulse_orb(cur, np.asarray(m.dv, dtype=np.float64), mu)
        u_cur = normalize_angle(cur.w + cur.nu)

    return cur
