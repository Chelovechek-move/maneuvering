import math
import numpy as np
from maneuvering.types import Scalar
from maneuvering.orbit.keplerian import KepMean, KepTrue
from maneuvering.orbit.utils import mean_motion
from maneuvering.orbit.convert_kep import calc_mean_from_true, calc_true_from_mean


def propagate_mean_anomaly(M: Scalar, a: Scalar, dt: Scalar, mu: Scalar) -> Scalar:
    """
    Делает прогноз средней аномалии по времени: M(t+Δt) = M + n·Δt, где n = √(μ / a³).

    Параметры
    ----------
    M : Scalar
        Текущая средняя аномалия, [рад].
    a : Scalar
        Большая полуось орбиты, [м].
    dt : Scalar
        Интервал прогноза по времени Δt, [с].
    mu : Scalar
        Гравитационный параметр центрального тела, [м³/с²].

    Возвращает
    -------
    Scalar
        Новая средняя аномалия в момент t+Δt (без нормализации), [рад].
    """
    return M + mean_motion(a, mu) * dt


def propagate_mean(o: KepMean, dt: Scalar, mu: Scalar) -> KepMean:
    """
    Делает прогноз средних кеплеровых элементов на интервал времени Δt.

    Параметры
    ----------
    o : KepMean
        Исходные средние элементы {a, e, w, i, raan, M} в момент t.
    dt : Scalar
        Интервал прогноза по времени Δt, [с].
    mu : Scalar
        Гравитационный параметр центрального тела, [м³/с²].

    Возвращает
    -------
    KepMean
        Те же элементы в момент t+Δt; отличаются только M, [рад].
    """
    new_M = propagate_mean_anomaly(o.M, o.a, dt, mu)
    return KepMean(a=o.a, e=o.e, w=o.w, i=o.i, raan=o.raan, M=new_M)


def propagate_true_anomaly(nu: Scalar, a: Scalar, e: Scalar, dt: Scalar, mu: Scalar) -> Scalar:
    """
    Делает прогноз истинной аномалии по времени.

    Параметры
    ----------
    nu : Scalar
        Текущая истинная аномалия, [рад].
    a : Scalar
        Большая полуось орбиты, [м].
    e : Scalar
        Эксцентриситет (0 ≤ e < 1), [1].
    dt : Scalar
        Интервал прогноза по времени Δt, [с].
    mu : Scalar
        Гравитационный параметр центрального тела, [м³/с²].

    Возвращает
    -------
    Scalar
        Новая истинная аномалия в момент t+Δt (с сохранением «целой части» 2π), [рад].
    """
    TWO_PI = 2.0 * math.pi

    # Выделяем целую часть и дробную
    main_nu = TWO_PI * math.floor(nu / TWO_PI)
    delta_nu = nu - main_nu

    # Средняя аномалия из истинной
    M = calc_mean_from_true(delta_nu, e)

    # Новая средняя аномалия
    new_M = propagate_mean_anomaly(M, a, dt, mu)

    # Разбиваем на целую и дробную
    main_new_M = TWO_PI * math.floor(new_M / TWO_PI)
    delta_new_M = new_M - main_new_M

    # Переводим обратно в истинную аномалию
    tol = 20 * np.finfo(float).eps
    new_nu = calc_true_from_mean(delta_new_M, e, max_iter=150, tol=tol)

    return new_nu + main_new_M + main_nu


def propagate_true(o: KepTrue, dt: Scalar, mu: Scalar) -> KepTrue:
    """
    Делает прогноз истинных кеплеровых элементов на интервал времени Δt.

    Параметры
    ----------
    o : KepTrue
        Исходные истинные элементы {a, e, w, i, raan, nu} в момент t.
    dt : Scalar
        Интервал прогноза по времени Δt, [с].
    mu : Scalar
        Гравитационный параметр центрального тела, [м³/с²].

    Возвращает
    -------
    KepTrue
        Те же элементы в момент t+Δt; отличаются только nu, [рад].
    """
    new_nu = propagate_true_anomaly(o.nu, o.a, o.e, dt, mu)
    return KepTrue(a=o.a, e=o.e, w=o.w, i=o.i, raan=o.raan, nu=new_nu)
