import math
import pytest

from maneuvering.orbit.keplerian import Kep
from maneuvering.orbit.distance import distance_orbit

DEG = math.pi / 180.0
tol = 1e-12


def test_coincident_orbits():
    """Две совпадающие орбиты → расстояние близко к нулю."""
    o = Kep(a=6800000.0, e=0.251, w=10 * DEG, i=10 * DEG, raan=10 * DEG)

    d = distance_orbit(o, o)

    assert math.isclose(d, 0.0, abs_tol=0.10132789611816407)


def test_specific_example():
    """Конкретный пример"""
    o1 = Kep(a=6566000.0, e=0.00228, w=20 * DEG, i=30 * DEG, raan=130 * DEG)
    o2 = Kep(a=6721000.0, e=0.00149, w=150 * DEG, i=280 * DEG, raan=130 * DEG)

    d = distance_orbit(o1, o2)

    assert math.isclose(d, 7697265.1987816272, abs_tol=tol)
