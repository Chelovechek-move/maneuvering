import math

from maneuvering import mean_motion


def test_mean_motion_basic():
    mu = 3.986004418e14  # Земля
    a = 7000e3
    n = mean_motion(mu, a)
    assert n > 0
    # проверка периода: period = 2*pi/n
    period = 2 * math.pi / n
    assert 5000 < period < 7000  # грубая проверка диапазона
