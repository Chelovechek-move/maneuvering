import numpy as np
import math

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


def solve_acos_bsin_c_eq_0(a: float, b: float, c: float, tol: float = 1e-12):
    """
    Решает a cos x + b sin x + c = 0 на [0, 2π).
    Возвращает список из 0, 1 или 2 решений (радианы, [0, 2π)).

    Особый случай: если |a|,|b| ≈ 0 и |c| ≈ 0, уравнение тождественно верно (беск. много решений) —
    функция возвращает пустой список (это оговорка интерфейса).
    """
    R = math.hypot(a, b)  # sqrt(a^2 + b^2)
    if R < tol:
        # уравнение вырождается в c = 0 (беск. решений) или не имеет решений
        return [] if abs(c) < tol else []

    # Приводим к R cos(x - φ) + c = 0
    phi = math.atan2(b, a)
    u = -c / R

    # Проверка достижимости и защита от округления
    if u > 1 + tol or u < -1 - tol:
        return []
    u = max(-1.0, min(1.0, u))

    y = math.acos(u)  # решения для t = x - φ:  t = ±y
    x1 = (y + phi) % (2 * math.pi)
    x2 = (-y + phi) % (2 * math.pi)

    # Удаляем возможный дубликат (когда решений фактически одно)
    diff = abs(x1 - x2)
    if diff < tol or abs(2 * math.pi - diff) < tol:
        return [x1 % (2 * math.pi)]
    else:
        xs = sorted([x1, x2])
        return xs
