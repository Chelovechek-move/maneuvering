import numpy as np

from maneuvering.types import Scalar, Vector3


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


def normalize_angle(a: Scalar) -> Scalar:
    """
    Нормализует угол в диапазон [0, 2π).

    Параметры
    ----------
    a : Scalar
        Угол, [рад].

    Возвращает
    -------
    Scalar
        Нормализованный угол в диапазоне [0, 2π), [рад].
    """
    twopi = 2.0 * np.pi
    x = float(a) % twopi
    return x + twopi if x < 0.0 else x
