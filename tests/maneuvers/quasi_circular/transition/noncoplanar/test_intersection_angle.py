# tests/test_planes_intersection_anomaly.py
from __future__ import annotations

import numpy as np

from maneuvering.maneuvers.quasi_circular.transition.noncoplanar.intersection_angle import (
    intersection_angle,
)
from maneuvering.orbit.keplerian import Kep

deg: float = np.pi / 180.0
tol: float = np.finfo(float).eps * 100.0


def test_intersection_angle_1():
    # Вектор эксцентриситета начальной орбиты сонаправлен с линией пересечения
    initial = Kep(a=7_000_000, e=0.0, w=0.0, i=0.0, raan=0.0)
    target = Kep(a=7_000_000, e=0.0, w=0.0, i=90.0 * deg, raan=0.0)
    expected = 0.0 * deg

    angle = intersection_angle(initial, target)
    assert abs(angle - expected) <= tol


def test_intersection_angle_2():
    # Эксцентриситет на 30° впереди линии пересечения
    initial = Kep(a=7_000_000, e=0.0, w=30.0 * deg, i=0.0, raan=0.0)
    target = Kep(a=7_000_000, e=0.0, w=0.0, i=90.0 * deg, raan=0.0)
    expected = 330.0 * deg

    angle = intersection_angle(initial, target)
    assert abs(angle - expected) <= tol


def test_intersection_angle_3():
    # Эксцентриситет на 60° впереди линии пересечения
    initial = Kep(a=7_000_000, e=0.0, w=60.0 * deg, i=0.0, raan=0.0)
    target = Kep(a=7_000_000, e=0.0, w=0.0, i=90.0 * deg, raan=0.0)
    expected = 300.0 * deg

    angle = intersection_angle(initial, target)
    assert abs(angle - expected) <= tol


def test_intersection_angle_4():
    # Эксцентриситет на 200° впереди линии пересечения
    initial = Kep(a=7_000_000, e=0.0, w=200.0 * deg, i=0.0, raan=0.0)
    target = Kep(a=7_000_000, e=0.0, w=0.0, i=90.0 * deg, raan=0.0)
    expected = 160.0 * deg

    angle = intersection_angle(initial, target)
    assert abs(angle - expected) <= tol


def test_intersection_angle_5():
    # Эксцентриситет на 330° впереди линии пересечения
    initial = Kep(a=7_000_000, e=0.0, w=330.0 * deg, i=0.0, raan=0.0)
    target = Kep(a=7_000_000, e=0.0, w=0.0, i=90.0 * deg, raan=0.0)
    expected = 30.0 * deg

    angle = intersection_angle(initial, target)
    assert abs(angle - expected) <= tol


def test_intersection_angle_6():
    # Более общий/сложный случай взаимного расположения орбит
    initial = Kep(a=7_000_000, e=0.15, w=200.0 * deg, i=4.0 * deg, raan=110.0 * deg)
    target = Kep(a=7_000_000, e=0.28, w=330.0 * deg, i=90.0 * deg, raan=80.0 * deg)
    expected = 129.9394539806205 * deg

    angle = intersection_angle(initial, target)
    assert abs(angle - expected) <= tol
