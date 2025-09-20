import numpy as np
from maneuvering.types import Vector3


def normalize(v: Vector3) -> Vector3:
    """
    Нормирует вектор: возвращает v / ||v||.

    Параметры
    ----------
    v : Vector3
        Входной 3D-вектор (float64).

    Возвращает
    -------
    Vector3
        Вектор той же формы с единичной нормой.
    """
    n = float(np.linalg.norm(v))
    return v / n
