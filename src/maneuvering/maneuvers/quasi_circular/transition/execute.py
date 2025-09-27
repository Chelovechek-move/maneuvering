from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from maneuvering.maneuvers.apply_impulse import apply_impulse_orb
from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.orbit.keplerian import KepTrue
from maneuvering.types import Scalar
from maneuvering.utils.math_tools import normalize_angle


def execute(oi: KepTrue, maneuvers: list[Maneuver], mu: Scalar) -> KepTrue:
    """
    Исполняет последовательность манёвров и возвращает финальную орбиту.

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
        cur = KepTrue(
            a=cur.a, e=cur.e, w=cur.w, i=cur.i, raan=cur.raan, nu=normalize_angle(cur.nu + du)
        )
        cur = apply_impulse_orb(cur, np.asarray(m.dv, dtype=np.float64), mu)
        u_cur = normalize_angle(cur.w + cur.nu)

    return cur


def execute_batch(
    oi: KepTrue,
    maneuvers: list[Maneuver],
    step: Scalar,
    mu: Scalar,
) -> list[KepTrue]:
    """
    Исполняет последовательность манёвров и возвращает требуемые промежуточные состояния.

    Идём по истинной широте u = w + nu вперёд, с постоянным шагом `step` (рад).
    Если ближайший манёвр расположен ближе, чем `step`, делаем точный “подскок” до манёвра,
    записываем состояние, применяем импульс, снова записываем состояние после манёвра,
    и продолжаем дальше. Возвращаем последовательность состояний `KepTrue`
    от начальной точки до момента последнего манёвра включительно.

    Parameters
    ----------
    oi : KepTrue
        Начальные истинные кеплеровы элементы.
    maneuvers : list[Maneuver]
        Список манёвров. Для каждого m:
          - m.angle — истинная широта u = w + nu, [рад];
          - m.dv    — импульс Δv в орбитальной СК {r, t, n}, [м/с].
    step : Scalar
        Шаг по истинной широте u (рад), > 0.
    mu : Scalar
        Гравитационный параметр [м^3/с^2].

    Returns
    -------
    list[KepTrue]
        Список состояний (копии): стартовое, все промежуточные шаги,
        состояния ровно в точках манёвров (до и после импульса), и финальное после последнего манёвра.

    Notes
    -----
    - Пассивное “продвижение” по орбите реализовано как увеличение `nu` на заданный угловой шаг
      (без расчёта времени), что удобно для построения графиков/треков.
    - Углы приводятся к [0, 2π).
    """
    two_pi = 2.0 * math.pi

    # Локальные утилиты
    def u(o: KepTrue) -> float:
        return normalize_angle(o.w + o.nu)

    def advance_by_du(o: KepTrue, du: float) -> KepTrue:
        return replace(o, nu=normalize_angle(o.nu + du))

    cur = replace(oi)
    out: list[KepTrue] = [cur]  # стартовая точка
    u_cur = u(cur)

    i = 0
    while i < len(maneuvers):
        m = maneuvers[i]
        du_to_m = (m.angle - u_cur) % two_pi

        if du_to_m <= step:
            # Подскок ровно до манёвра (если не уже там)
            if du_to_m > 0.0:
                cur = advance_by_du(cur, du_to_m)
                out.append(cur)  # состояние ДО импульса в точке манёвра

            # Применяем импульс в орбитальной СК
            cur = apply_impulse_orb(cur, m.dv, mu)
            out.append(cur)  # состояние ПОСЛЕ импульса

            u_cur = u(cur)
            i += 1
        else:
            # Обычный шаг по широте
            cur = advance_by_du(cur, step)
            out.append(cur)
            u_cur = u(cur)

    return out
