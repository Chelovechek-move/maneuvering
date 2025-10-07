from __future__ import annotations

import numpy as np

from maneuvering.types import Scalar


def normal_vector(i: Scalar, raan: Scalar) -> np.ndarray:
    """
    Рассчитывает единичный вектор нормали к плоскости орбиты.

    Параметры
    ----------
    inclination : Scalar
        Наклонение орбиты, [рад].
    raan : Scalar
        Долгота восходящего узла, [рад].

    Возвращает
    ----------
    np.ndarray shape (3,)
        Единичный вектор нормали к плоскости орбиты (в ECI): (sin i sin Ω, -sin i cos Ω, cos i).
    """
    sin_i = float(np.sin(i))
    cos_i = float(np.cos(i))
    sin_raan = float(np.sin(raan))
    cos_raan = float(np.cos(raan))
    return np.array([sin_i * sin_raan, -sin_i * cos_raan, cos_i], dtype=float)
