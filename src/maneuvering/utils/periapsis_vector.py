from __future__ import annotations

import math
import numpy as np

from maneuvering.orbit.keplerian import Kep


def periapsis_vector(o: Kep) -> np.ndarray:
    """
    Вектор перицентра по кеплеровым элементам (радианы).
    Возвращает вектор p_hat:
      p_hat : np.ndarray shape (3,) — единичный вектор на перицентр в ИСК

    Параметры
    ---------
    a    : большая полуось (м)
    e    : эксцентриситет
    i    : наклонение, рад
    raan : долгота восходящего узла Ω, рад
    w    : аргумент перицентра ω, рад
    """
    cos_raan, sin_raan = math.cos(o.raan), math.sin(o.raan)
    cos_i, sin_i = math.cos(o.i),    math.sin(o.i)
    cos_w, sin_w = math.cos(o.w),    math.sin(o.w)

    # Единичный вектор на перицентр (перефокальная ось P, перенесённая в ИСК)
    p_hat = np.array([
        cos_raan*cos_w - sin_raan*sin_w*cos_i,
        sin_raan*cos_w + cos_raan*sin_w*cos_i,
        sin_w*sin_i
    ], dtype=float)

    return p_hat
