from maneuvering import solve_lambert


def test_solve_lambert_stub():
    dv1, dv2 = solve_lambert([1, 0, 0], [0, 1, 0], 1000.0, 3.986e14)
    assert dv1 == 0.0 and dv2 == 0.0
