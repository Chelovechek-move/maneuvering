import math
from maneuvering.types import Scalar, Vector3


def mean_motion(a: Scalar, mu: Scalar) -> Scalar:
    """
    Среднее движение (mean motion).

    Формула: n = sqrt(mu / a^3)

    Параметры
    ---------
    a : Scalar
        Большая полуось, [м].
    mu : Scalar
        Гравитационный параметр, [м^3/с^2].

    Возвращает
    ----------
    n : Scalar
        Среднее движение, [рад/с].
    """
    return math.sqrt(mu / a) / a


def period(a: Scalar, mu: Scalar) -> Scalar:
    """
    Период орбиты.

    Формула: T = 2π * sqrt(a^3 / mu)

    Параметры
    ---------
    a : Scalar
        Большая полуось, [м].
    mu : Scalar
        Гравитационный параметр, [м^3/с^2].

    Возвращает
    ----------
    T : Scalar
        Период орбиты, [с].
    """
    return 2.0 * math.pi / mean_motion(a, mu)
