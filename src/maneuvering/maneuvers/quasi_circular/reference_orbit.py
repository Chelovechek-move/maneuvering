from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from maneuvering.types import Scalar
from maneuvering.orbit.keplerian import Kep


@dataclass(frozen=True, slots=True)
class RefOrbit:
    """
    Параметры опорной круговой орбиты.

    Атрибуты
    --------
    r : Scalar
        Радиус опорной круговой орбиты, [м].
    v : Scalar
        Круговая скорость: v = sqrt(μ / r), [м/с].
    """
    r: Scalar
    v: Scalar


@dataclass(frozen=True, slots=True)
class TransDevs:
    """
    Отклонения от опорной круговой орбиты (для задачи перехода).

    Атрибуты
    --------
    ex : Scalar
        ΔE_x = e_t·cos(ω_t) − e_i·cos(ω_i), [1].
    ey : Scalar
        ΔE_y = e_t·sin(ω_t) − e_i·sin(ω_i), [1].
    e : Scalar
        ΔE = sqrt(ΔE_x² + ΔE_y²), [1].
    a : Scalar
        ΔA = (a_t − a_i) / a_ref, [1].
    i : Scalar
        Ориентированный угол между плоскостями орбит, [рад].
    """
    ex: Scalar
    ey: Scalar
    e: Scalar
    a: Scalar
    i: Scalar


def reference_orbit(oi: Kep, ot: Kep, mu: Scalar) -> RefOrbit:
    """
    Рассчитывает параметры опорной круговой орбиты (радиус и круговую скорость).

    Формулы
    -------
    r = (a_i + a_t) / 2
    v = sqrt(μ / r)

    Параметры
    ----------
    oi : Kep
        Начальная орбита {a [м], e [1], w [рад], i [рад], raan [рад]}.
    ot : Kep
        Целевая орбита {a [м], e [1], w [рад], i [рад], raan [рад]}.
    mu : Scalar
        Гравитационный параметр центрального тела μ, [м³/с²].

    Возвращает
    -------
    RefOrbit
        Параметры опорной круговой орбиты: r [м], v [м/с].
    """
    radius = (oi.a + ot.a) * 0.5
    velocity = float(np.sqrt(mu / radius))
    return RefOrbit(r=radius, v=velocity)


def trans_devs(oi: Kep, ot: Kep) -> TransDevs:
    """
    Рассчитывает отклонения от опорной круговой орбиты для пары орбит.

    Формулы
    -------
    ΔE_x = e_t·cos(ω_t) − e_i·cos(ω_i)
    ΔE_y = e_t·sin(ω_t) − e_i·sin(ω_i)
    ΔE   = ||Δe⃗|| = sqrt(ΔE_x² + ΔE_y²)

    a_ref = (a_i + a_t) / 2
    ΔA = (a_t − a_i) / a_ref

    ΔΩ = Ω_t − Ω_i
    cos Φ = cos i_t · cos i_i + sin i_t · sin i_i · cos(ΔΩ)
    Δi = sign · 2 · sin( arccos(clamp(cos Φ, −1, 1)) / 2 ),
      sign = +1, если i_t > i_i; иначе −1.

    Параметры
    ----------
    oi : Kep
        Начальная орбита {a [м], e [1], w [рад], i [рад], raan [рад]}.
    ot : Kep
        Целевая орбита {a [м], e [1], w [рад], i [рад], raan [рад]}.

    Возвращает
    -------
    TransDevs
        Отклонения ΔE_x, ΔE_y, ΔE, ΔA и ориентированный угол между плоскостями орбит [рад].
    """
    ex_i = oi.e * float(np.cos(oi.w))
    ey_i = oi.e * float(np.sin(oi.w))
    ex_t = ot.e * float(np.cos(ot.w))
    ey_t = ot.e * float(np.sin(ot.w))

    d_ex = ex_t - ex_i
    d_ey = ey_t - ey_i
    d_e = float(np.hypot(d_ex, d_ey))

    a_ref = (oi.a + ot.a) * 0.5
    d_a = (ot.a - oi.a) / a_ref

    d_raan = ot.raan - oi.raan
    cos_phi = (
        float(np.cos(ot.i)) * float(np.cos(oi.i))
        + float(np.sin(ot.i)) * float(np.sin(oi.i)) * float(np.cos(d_raan))
    )

    # Численно-устойчивый arccosN: зажимаем аргумент в [-1, 1]
    cos_phi = float(np.clip(cos_phi, -1.0, 1.0))
    half_angle = 0.5 * float(np.arccos(cos_phi))

    sign = 1.0 if (ot.i > oi.i) else -1.0
    angle = sign * 2.0 * float(np.sin(half_angle))

    return TransDevs(
        ex=d_ex,
        ey=d_ey,
        e=d_e,
        a=d_a,
        i=angle,
    )
