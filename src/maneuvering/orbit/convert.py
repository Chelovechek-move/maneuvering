from __future__ import annotations

import numpy as np

from maneuvering.orbit.cartesian import Cart
from maneuvering.orbit.keplerian import KepTrue, kep_true
from maneuvering.types import Scalar, Vector3


def convert_kep_true_to_cart(o: KepTrue, mu: Scalar) -> Cart:
    """
    Конвертирует истинные кеплеровы элементы в декартовы элементы орбиты.

    Параметры
    ----------
    o : KepTrue
        {a, e, w, i, raan, nu}.
    mu : Scalar
        Гравитационный параметр, [м^3/с^2].

    Возвращает
    -------
    Cart
        {r [м], v [м/с]}.
    """
    a = o.orb.a
    e = o.orb.e
    w = o.orb.w
    i = o.orb.i
    raan = o.orb.raan
    nu = o.nu

    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(nu))

    cos_u = np.cos(w + nu)
    sin_u = np.sin(w + nu)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)

    x = (cos_u * cos_raan - cos_i * sin_u * sin_raan) * r_mag
    y = (cos_u * sin_raan + cos_i * sin_u * cos_raan) * r_mag
    z = (sin_u * sin_i) * r_mag
    r: Vector3 = np.array([x, y, z], dtype=np.float64)

    sqrt_mu_p = np.sqrt(mu / p)
    cos_nu_p_e = cos_nu + e
    cos_w = np.cos(w)
    sin_w = np.sin(w)

    vx = sqrt_mu_p * cos_nu_p_e * (
        -sin_w * cos_raan - cos_i * sin_raan * cos_w
    ) - sqrt_mu_p * sin_nu * (cos_w * cos_raan - cos_i * sin_raan * sin_w)
    vy = sqrt_mu_p * cos_nu_p_e * (
        -sin_w * sin_raan + cos_i * cos_raan * cos_w
    ) - sqrt_mu_p * sin_nu * (cos_w * sin_raan + cos_i * cos_raan * sin_w)
    vz = sqrt_mu_p * (cos_nu_p_e * sin_i * cos_w - sin_nu * sin_i * sin_w)

    v: Vector3 = np.array([vx, vy, vz], dtype=np.float64)

    return Cart(r=r, v=v)


def convert_cart_to_kep_true(o: Cart, mu: Scalar) -> KepTrue:
    """
    Конвертирует декартовы элементы в истинные кеплеровы элементы орбиты.

    Параметры
    ----------
    orbit : Cart
        {r [м], v [м/с]}.
    mu : Scalar
        Гравитационный параметр, [м^3/с^2].

    Возвращает
    -------
    KeplerianTrue
        {a, e, w, i, raan, nu}.
    """
    r = o.r
    v = o.v

    def norm(x):
        return float(np.linalg.norm(x))

    def hat(x):
        n = norm(x)
        return x / n if n > 0.0 else x  # единичный вектор, если ненулевой

    def dot(a, b):
        return float(np.dot(a, b))

    cross = np.cross
    atan2 = np.arctan2
    two_pi = 2.0 * np.pi

    def normalize_ang(ang):
        return (ang % two_pi + two_pi) % two_pi  # в [0, 2π)

    v2 = dot(v, v)
    r_norm = norm(r)

    h = cross(r, v)  # орбитальный момент
    h_hat = hat(h)
    node = cross(np.array([0.0, 0.0, 1.0]), h)  # вектор линии узлов
    node_norm = norm(node)

    e1 = (node / node_norm) if node_norm > 0.0 else np.array([1.0, 0.0, 0.0])
    e2 = cross(h_hat, e1)

    # эксцентриситет-вектор
    ecc_vec = ((v2 - mu / r_norm) * r - dot(r, v) * v) / mu
    e = norm(ecc_vec)

    g1 = (ecc_vec / e) if e > 0.0 else e1
    g2 = cross(h_hat, g1)

    energy = v2 * 0.5 - mu / r_norm
    a = -mu / (2.0 * energy)

    vec_for_inc = cross(e1, np.array([0.0, 0.0, 1.0]))
    inc_x = h[2]
    inc_y = dot(h, vec_for_inc)
    i = atan2(inc_y, inc_x)

    raan = atan2(e1[1], e1[0])

    ex = dot(g1, e1)
    ey = dot(g1, e2)
    w = atan2(ey, ex)

    pos_x = dot(r, g1)
    pos_y = dot(r, g2)
    nu = atan2(pos_y, pos_x)

    return kep_true(
        a=a,
        e=e,
        w=normalize_ang(w),
        i=i,
        raan=normalize_ang(raan),
        nu=normalize_ang(nu),
    )
