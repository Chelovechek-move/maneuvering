from __future__ import annotations

import math
import numpy as np

from maneuvering.types import Scalar
from maneuvering.utils.math_tools import normalize_angle


def calc_mean_from_eccentric(E: Scalar, e: Scalar) -> Scalar:
    """
    Рассчитывает среднюю аномалию из эксцентрической: M = E − e·sin(E).

    Параметры
    ----------
    E : Scalar
        Эксцентрическая аномалия, [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].

    Возвращает
    -------
    Scalar
        Средняя аномалия, нормализованная в диапазон [0, 2π), [рад].
    """
    E_n = normalize_angle(E)
    M = E_n - e * math.sin(E_n)
    return normalize_angle(M)


def calc_eccentric_from_true(nu: Scalar, e: Scalar) -> Scalar:
    """
    Рассчитывает эксцентрическую аномалию из истинной.

    Параметры
    ----------
    nu : Scalar
        Истинная аномалия, [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].

    Возвращает
    -------
    Scalar
        Эксцентрическая аномалия в диапазоне [0, 2π), [рад].
    """
    s = math.sin(nu) * math.sqrt(1.0 - e * e)
    c = e + math.cos(nu)
    E = math.atan2(s, c)
    return E if E >= 0.0 else (E + 2.0 * math.pi)


def calc_eccentric_from_mean(M: Scalar, e: Scalar, max_newton_iter: int = 150,
                             tol: Scalar = np.finfo(float).eps * 100.0) -> Scalar:
    """
    Рассчитывает эксцентрическую аномалию из средней (решение уравнения Кеплера методом Ньютона).

    Параметры
    ----------
    M : Scalar
        Средняя аномалия, [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].
    max_newton_iter : int
        Максимальное число итераций метода Ньютона, [1].
    tol : Scalar
        Критерий остановки по |ΔE|, [рад].

    Возвращает
    -------
    Scalar
        Эксцентрическая аномалия (без финальной нормализации), [рад].
    """
    M_n = normalize_angle(M)
    E = (M_n - e) if (M_n > math.pi) else (M_n + e)
    for _ in range(int(max_newton_iter)):
        sin_E = math.sin(E)
        cos_E = math.cos(E)
        delta = (M_n + e * sin_E - E) / (1.0 - e * cos_E)
        E += delta
        if abs(delta) < tol:
            return E
    return E


def calc_true_from_eccentric(E: Scalar, e: Scalar) -> Scalar:
    """
    Рассчитывает истинную аномалию из эксцентрической.

    Параметры
    ----------
    E : Scalar
        Эксцентрическая аномалия, [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].

    Возвращает
    -------
    Scalar
        Истинная аномалия, нормализованная в диапазон [0, 2π), [рад].
    """
    s = math.sin(E) * math.sqrt(1.0 - e * e)
    c = math.cos(E) - e
    nu = math.atan2(s, c)
    return nu if nu >= 0.0 else (nu + 2.0 * math.pi)


def calc_true_from_mean(M: Scalar, e: Scalar, max_iter: int = 150, tol: Scalar = np.finfo(float).eps * 100.0) -> Scalar:
    """
    Рассчитывает истинную аномалию из средней: M → E → ν.

    Параметры
    ----------
    M : Scalar
        Средняя аномалия, [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].
    max_iter : int
        Максимальное число итераций метода Ньютона, [1].
    tol : Scalar
        Критерий остановки для расчёта E, [рад].

    Возвращает
    -------
    Scalar
        Истинная аномалия, нормализованная в диапазон [0, 2π), [рад].
    """
    E = calc_eccentric_from_mean(M, e, max_iter, tol)
    return normalize_angle(calc_true_from_eccentric(E, e))


def calc_mean_from_true(nu: Scalar, e: Scalar) -> Scalar:
    """
    Рассчитывает среднюю аномалию из истинной: ν → E → M.

    Параметры
    ----------
    nu : Scalar
        Истинная аномалия, [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].

    Возвращает
    -------
    Scalar
        Средняя аномалия, нормализованная в диапазон [0, 2π), [рад].
    """
    E = calc_eccentric_from_true(nu, e)
    return normalize_angle(calc_mean_from_eccentric(E, e))


def calc_true_from_mean_non_norm(M: Scalar, e: Scalar, max_iter: int = 150,
                                 tol: Scalar = np.finfo(float).eps * 100.0) -> Scalar:
    """
    Рассчитывает истинную аномалию из средней без нормализации.

    Параметры
    ----------
    M : Scalar
        Средняя аномалия (может включать «целую часть» 2π), [рад].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].
    max_iter : int
        Максимальное число итераций метода Ньютона, [1].
    tol : Scalar
        Критерий остановки для расчёта E, [рад].

    Возвращает
    -------
    Scalar
        Истинная аномалия с сохранением «целой части», [рад].
    """
    frac = normalize_angle(M)
    whole = M - frac
    return whole + calc_true_from_mean(frac, e, max_iter, tol)
