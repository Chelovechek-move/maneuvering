from __future__ import annotations

import math

import numpy as np

from maneuvering.orbit.keplerian import Kep
from maneuvering.types import Scalar, Vector3


def _calc_P(o: Kep) -> Vector3:
    """
    Рассчитывает вектор P орбитальной плоскости.

    Параметры
    ----------
    o : Kep
        Кеплеровы элементы {a [м], e [1], w [рад], i [рад], raan [рад]}.

    Возвращает
    -------
    Vector3
        Вектор P в ECI, [1].
    """
    cos_w = math.cos(o.w)
    sin_w = math.sin(o.w)
    cos_i = math.cos(o.i)
    sin_i = math.sin(o.i)
    cos_raan = math.cos(o.raan)
    sin_raan = math.sin(o.raan)

    # P = (cos_w*cos_raan - cos_i*sin_w*sin_raan, cos_w*sin_raan + cos_i*sin_w*cos_raan, sin_i*sin_w)
    return np.array(
        [
            cos_w * cos_raan - cos_i * sin_w * sin_raan,
            cos_w * sin_raan + cos_i * sin_w * cos_raan,
            sin_i * sin_w,
        ],
        dtype=np.float64,
    )


def _calc_Q(o: Kep) -> Vector3:
    """
    Рассчитывает вектор Q орбитальной плоскости.

    Параметры
    ----------
    o : Kep
        Кеплеровы элементы {a [м], e [1], w [рад], i [рад], raan [рад]}.

    Возвращает
    -------
    Vector3
        Вектор Q в ECI, [1].
    """
    cos_w = math.cos(o.w)
    sin_w = math.sin(o.w)
    cos_i = math.cos(o.i)
    sin_i = math.sin(o.i)
    cos_raan = math.cos(o.raan)
    sin_raan = math.sin(o.raan)

    # Q = (-sin_w*cos_raan - cos_i*cos_w*sin_raan, -sin_w*sin_raan + cos_i*cos_w*cos_raan, sin_i*cos_w)
    return np.array(
        [
            -sin_w * cos_raan - cos_i * cos_w * sin_raan,
            -sin_w * sin_raan + cos_i * cos_w * cos_raan,
            sin_i * cos_w,
        ],
        dtype=np.float64,
    )


def distance_orbit(o1: Kep, o2: Kep) -> Scalar:
    """
    Рассчитывает расстояние между двумя эллиптическими орбитами.

    Параметры
    ----------
    o1, o2 : Kep
        Кеплеровы элементы {a [м], e [1], w [рад], i [рад], raan [рад]} без аномалии.

    Возвращает
    -------
    Scalar
        Метрика расстояния между орбитами, [м].

    Notes
    -----
    - Данная метрика и все используемые обозначения полностью соответствуют статье Холшевникова К. В.
    https://doi.org/10.1023/B:CELE.0000034504.41897.ac
    """

    def sqr(x: Scalar) -> Scalar:
        return x * x

    P1 = _calc_P(o1)
    Q1 = _calc_Q(o1)
    P2 = _calc_P(o2)
    Q2 = _calc_Q(o2)

    eta1 = math.sqrt(1.0 - sqr(o1.e))
    eta2 = math.sqrt(1.0 - sqr(o2.e))

    S1 = eta1 * Q1
    S2 = eta2 * Q2

    alpha1 = o1.a / o2.a
    alpha2 = o2.a / o1.a

    P1P2 = float(np.dot(P1, P2))
    P1S2 = float(np.dot(P1, S2))
    P2S1 = float(np.dot(P2, S1))
    S1S2 = float(np.dot(S1, S2))

    W0 = (
        2.0 * (alpha1 + alpha2) + alpha1 * sqr(o1.e) + alpha2 * sqr(o2.e) - 4.0 * P1P2 * o1.e * o2.e
    ) / 4.0
    W5 = -P1P2 / 2.0
    W6 = -P1S2 / 2.0
    W7 = -P2S1 / 2.0
    W8 = -S1S2 / 2.0

    inner = W0 - math.sqrt(sqr(W5 + W8) + sqr(W6 - W7))
    inner = max(inner, 0.0)  # защита от отрицательного нуля/ошибок округления

    return math.sqrt(2.0 * o1.a * o2.a * inner)
