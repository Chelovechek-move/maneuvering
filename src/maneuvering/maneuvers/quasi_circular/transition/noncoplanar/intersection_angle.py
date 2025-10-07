from __future__ import annotations

import numpy as np

from maneuvering.orbit.keplerian import Kep
from maneuvering.utils.math_tools import normalize_angle
from maneuvering.utils.normal_vector import normal_vector


def intersection_angle(oi: Kep, ot: Kep) -> float:
    """
    Истинная аномалия линии пересечения плоскостей орбит на начальной орбите (некомпланарный случай).

    Параметры
    ---------
    oi : Kep
        Начальная орбита {a, e, w, i, raan}.
    ot : Kep
        Целевая орбита {a, e, w, i, raan}.

    Возвращает
    ----------
    float
        Угол между вектором эксцентриситета начальной орбиты и линией пересечения плоскостей,
        нормализован в [0, 2π).

    Принято
    -------
    Положительное направление против часовой стрелки.
    """
    # Нормали к плоскостям
    n1 = normal_vector(oi.i, oi.raan)
    n2 = normal_vector(ot.i, ot.raan)

    # Линия пересечения плоскостей (ненормированный)
    n_tmp = np.cross(n1, n2)

    # Восходящий узел начальной орбиты
    k = np.array([0.0, 0.0, 1.0])
    asc_vec = np.cross(k, n1)
    asc_norm = np.linalg.norm(asc_vec)
    l1 = asc_vec / asc_norm if asc_norm != 0.0 else np.array([1.0, 0.0, 0.0])

    # Ортогональ к l1 в плоскости орбиты
    l2 = np.cross(n1, l1)

    # Направление на перицентр начальной орбиты и орт к нему
    cw, sw = np.cos(oi.w), np.sin(oi.w)
    e1 = l1 * cw + l2 * sw
    e2 = np.cross(n1, e1)

    # Выбрать направление линии узлов, ближнее к l1
    sign = 1.0 if float(np.dot(n_tmp, l1)) >= 0.0 else -1.0
    n = sign * n_tmp
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        # Орбиты компланарны: линия пересечения неопределена; вернуть 0
        return 0.0
    n /= n_norm

    # Истинная аномалия линии пересечения в базисе (e1, e2)
    phi = float(np.arctan2(np.dot(e2, n), np.dot(e1, n)))
    return float(normalize_angle(phi))
