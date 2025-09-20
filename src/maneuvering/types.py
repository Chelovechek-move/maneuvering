import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

Scalar: TypeAlias = float

ArrayF64: TypeAlias = NDArray[np.float64]

Vector3: TypeAlias = ArrayF64
VectorN: TypeAlias = ArrayF64

def as_vector3(x: ArrayF64 | list[float] | tuple[float, float, float], *, copy: bool = False) -> Vector3:
    """
    Приводит вход к np.ndarray[np.float64] формы (3,).
    Упадёт с ValueError, если длина не равна 3.
    """
    arr = np.array(x, dtype=np.float64, copy=copy)
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape}")
    return arr
