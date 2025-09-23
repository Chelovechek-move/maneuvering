import math

import numpy as np

from maneuvering.maneuvers.apply_impulse import apply_impulse_eci, apply_impulse_orb
from maneuvering.orbit.convert_kep_cart import convert_kep_true_to_cart
from maneuvering.orbit.keplerian import KepTrue
from maneuvering.orbit.orbital_system import calc_orb_sys, rot_mat_orb_to_eci

mu = 3.9860044158e14
DEG = math.pi / 180.0

TOL_EQ_A = 1e-8  # м
TOL_EQ_E = 1e-12
TOL_EQ_ANG = 1e-12  # рад


def _assert_orbits_close(o1: KepTrue, o2: KepTrue, ta=TOL_EQ_A, te=TOL_EQ_E, tang=TOL_EQ_ANG):
    assert abs(o1.a - o2.a) <= ta
    assert abs(o1.e - o2.e) <= te
    assert abs(o1.w - o2.w) <= tang
    assert abs(o1.i - o2.i) <= tang
    assert abs(o1.raan - o2.raan) <= tang
    assert abs(o1.nu - o2.nu) <= tang


def test_equivalence_orb_vs_eci_any_state():
    """apply_impulse_orb эквивалентен apply_impulse_eci с правильно повернутым импульсом."""
    o = KepTrue(a=7200e3, e=0.01, w=30 * DEG, i=50 * DEG, raan=200 * DEG, nu=1.0)

    # Орбитальный импульс (произвольный)
    imp_orb = np.array([0.2, -0.5, 0.1], dtype=np.float64)

    # Поворот в ECI
    cart = convert_kep_true_to_cart(o, mu)
    R = rot_mat_orb_to_eci(calc_orb_sys(cart.r, cart.v))
    imp_eci = R @ imp_orb

    o1 = apply_impulse_orb(o, imp_orb, mu)
    o2 = apply_impulse_eci(o, imp_eci, mu)

    _assert_orbits_close(o1, o2)


def test_reversibility_same_point():
    """Подаём +Δv, затем -Δv в той же точке — возвращаемся к исходной орбите."""
    o = KepTrue(a=7300e3, e=0.01, w=40 * DEG, i=25 * DEG, raan=60 * DEG, nu=1.2)
    imp_eci = np.array([0.3, -0.4, 0.2], dtype=np.float64)

    o1 = apply_impulse_eci(o, imp_eci, mu)
    o2 = apply_impulse_eci(o1, -imp_eci, mu)
    _assert_orbits_close(o, o2)
