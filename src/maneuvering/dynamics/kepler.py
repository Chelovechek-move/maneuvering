import math


def mean_motion(mu: float, a: float) -> float:
    """
    Среднее движение n = sqrt(mu / a^3).
    mu — гравитационный параметр, a — большая полуось (м).
    """
    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")
    return math.sqrt(mu / (a**3))
