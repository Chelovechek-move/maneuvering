import math

import numpy as np

from maneuvering.orbit.convert import (
    convert_cart_to_kep_true,
    convert_kep_true_to_cart,
)
from maneuvering.orbit.keplerian import kep_true

MU = 3.9860044158e14

EPS = np.finfo(float).eps

ATOL = EPS * 10000
RTOL = EPS * 100


def _norm_angle(a: float) -> float:
    """Нормализация угла в [0, 2π)."""
    twopi = 2.0 * math.pi
    x = a % twopi
    return x + twopi if x < 0.0 else x


def _is_ang_close(a: float, b: float, atol: float = ATOL) -> bool:
    """Сравнение углов по модулю 2π с машинной точностью."""
    da = _norm_angle(a) - _norm_angle(b)
    # приведём разницу к [-π, π]
    da = (da + math.pi) % (2.0 * math.pi) - math.pi
    return abs(da) <= atol


def test_convert_kepler_true_cart():
    """
    Сетка по элементам:
      a   — несколько типичных орбит,
      e   — от слабой эллиптичности до высокой, но избегаем e≈0,
      i   — наклонения, избегаем i≈0,
      ω,Ω,ν — несколько углов на окружности.

    Для каждого набора: Kepler(True) -> Cartesian -> Kepler(True)
    и сравнение всех 6 параметров.
    """
    a_list = [7_000e3, 12_000e3, 42_164e3]
    e_list = [0.001, 0.2, 0.7]
    i_list = [0.2, 1.0, 1.7]
    w_list = [0.1, 1.3, 2.7]
    raan_list = [0.2, 2.1, 5.5]
    nu_list = [0.05, 1.0, 2.0, 3.1, 5.9]

    for a in a_list:
        for e in e_list:
            for inc in i_list:
                for w in w_list:
                    for raan in raan_list:
                        for nu in nu_list:
                            k_old = kep_true(a=a, e=e, w=w, i=inc, raan=raan, nu=nu)

                            cart = convert_kep_true_to_cart(k_old, MU)
                            k_new = convert_cart_to_kep_true(cart, MU)

                            # Скалярные параметры
                            assert math.isclose(k_new.a, a, rel_tol=RTOL, abs_tol=ATOL)
                            assert math.isclose(k_new.e, e, rel_tol=RTOL, abs_tol=ATOL)
                            assert math.isclose(k_new.i, inc, rel_tol=RTOL, abs_tol=ATOL)

                            # Углы: сравниваем с учётом эквивалентности по 2π
                            assert _is_ang_close(k_new.w, w)
                            assert _is_ang_close(k_new.raan, raan)
                            assert _is_ang_close(k_new.nu, nu)


def test_convert_random_samples(seed: int = 123):
    """
    Дополнительная случайная выборка — страхуемся от «сеточных совпадений».
    Избегаем сингулярностей, ограничивая e и i подальше от нуля.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        a = rng.uniform(6_800e3, 50_000e3)
        e = rng.uniform(0.001, 0.8)
        inc = rng.uniform(0.05, math.pi - 0.05)  # не около 0/π
        w = rng.uniform(0.0, 2.0 * math.pi)
        raan = rng.uniform(0.0, 2.0 * math.pi)
        nu = rng.uniform(0.0, 2.0 * math.pi)

        k_old = kep_true(a=a, e=e, w=w, i=inc, raan=raan, nu=nu)

        cart = convert_kep_true_to_cart(k_old, MU)
        kt2 = convert_cart_to_kep_true(cart, MU)

        assert math.isclose(kt2.a, a, rel_tol=RTOL, abs_tol=ATOL)
        assert math.isclose(kt2.e, e, rel_tol=RTOL, abs_tol=ATOL)
        assert math.isclose(kt2.i, inc, rel_tol=RTOL, abs_tol=ATOL)
        assert _is_ang_close(kt2.w, w)
        assert _is_ang_close(kt2.raan, raan)
        assert _is_ang_close(kt2.nu, nu)


RTOL_POS = 10 * EPS
RTOL_VEL = 10 * EPS
RTOL_BACK = 100 * EPS

pi = math.pi


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """||a-b|| / ||b|| (безопасно для нулевого b)."""
    na = float(np.linalg.norm(a - b))
    nb = float(np.linalg.norm(b))
    return na / nb if nb != 0.0 else na  # если b==0, сравниваем абсолютную норму


def _case(a, e, i, w, raan, nu, ref_r, ref_v):
    """
    helper: строит орбиту и сравнивает с эталоном позицию/скорость,
    затем делает круг обратно и сверяет относительную ошибку.
    """
    k_old = kep_true(a=a, e=e, w=w, i=i, raan=raan, nu=nu)

    cart = convert_kep_true_to_cart(k_old, MU)

    # проверка позиций и скоростей с относительной точностью как в C++-тестах
    r_err = _rel_err(cart.r, ref_r)
    v_err = _rel_err(cart.v, ref_v)
    assert r_err <= RTOL_POS, f"pos rel.err={r_err:.3e} > {RTOL_POS:.3e}"
    assert v_err <= RTOL_VEL, f"vel rel.err={v_err:.3e} > {RTOL_VEL:.3e}"

    # обратный круг: cart -> kep -> cart
    kt2 = convert_cart_to_kep_true(cart, MU)
    cart2 = convert_kep_true_to_cart(kt2, MU)

    r_back = _rel_err(cart2.r, cart.r)
    v_back = _rel_err(cart2.v, cart.v)
    assert r_back <= RTOL_BACK
    assert v_back <= RTOL_BACK


def test_singular_cases():
    a = 6800e3
    e = 0.0

    # скорость при круговой орбите (|r| = a)
    spd = math.sqrt(MU / a)

    # 1) i=0, w=0, raan=0, nu=0
    _case(a, e, 0.0, 0.0, 0.0, 0.0, np.array([a, 0.0, 0.0]), np.array([0.0, spd, 0.0]))

    # 2) i=0, w=0, raan=0, nu=π/2
    _case(a, e, 0.0, 0.0, 0.0, pi / 2, np.array([0.0, a, 0.0]), np.array([-spd, 0.0, 0.0]))

    # 3) i=0, w=0, raan=0, nu=π
    _case(a, e, 0.0, 0.0, 0.0, pi, np.array([-a, 0.0, 0.0]), np.array([0.0, -spd, 0.0]))

    # 4) i=0, w=0, raan=0, nu=3π/2
    _case(a, e, 0.0, 0.0, 0.0, 3 * pi / 2, np.array([0.0, -a, 0.0]), np.array([spd, 0.0, 0.0]))

    # 5) i=π (ретроградная), w=0, raan=0, nu=0
    _case(a, e, pi, 0.0, 0.0, 0.0, np.array([a, 0.0, 0.0]), np.array([0.0, -spd, 0.0]))

    # 6) i=π, w=0, raan=0, nu=π/2
    _case(a, e, pi, 0.0, 0.0, pi / 2, np.array([0.0, -a, 0.0]), np.array([-spd, 0.0, 0.0]))

    # 7) i=π, w=0, raan=0, nu=π
    _case(a, e, pi, 0.0, 0.0, pi, np.array([-a, 0.0, 0.0]), np.array([0.0, spd, 0.0]))

    # 8) i=π, w=0, raan=0, nu=3π/2
    _case(a, e, pi, 0.0, 0.0, 3 * pi / 2, np.array([0.0, a, 0.0]), np.array([spd, 0.0, 0.0]))

    # 9) i=π/2 (полярная), w=0, raan=0, nu=0
    _case(a, e, pi / 2, 0.0, 0.0, 0.0, np.array([a, 0.0, 0.0]), np.array([0.0, 0.0, spd]))

    # 10) i=π/2, w=0, raan=0, nu=π/2
    _case(a, e, pi / 2, 0.0, 0.0, pi / 2, np.array([0.0, 0.0, a]), np.array([-spd, 0.0, 0.0]))

    # 11) i=π/2, w=0, raan=0, nu=π
    _case(a, e, pi / 2, 0.0, 0.0, pi, np.array([-a, 0.0, 0.0]), np.array([0.0, 0.0, -spd]))

    # 12) i=π/2, w=0, raan=π, nu=0
    _case(a, e, pi / 2, 0.0, pi, 0.0, np.array([-a, 0.0, 0.0]), np.array([0.0, 0.0, spd]))

    # 13) i=π/2, w=π/2, raan=π, nu=π/2
    _case(a, e, pi / 2, pi / 2, pi, pi / 2, np.array([a, 0.0, 0.0]), np.array([0.0, 0.0, -spd]))
