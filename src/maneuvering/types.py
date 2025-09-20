from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Scalar: TypeAlias = float

ArrayF64: TypeAlias = NDArray[np.float64]

Vector3: TypeAlias = ArrayF64
VectorN: TypeAlias = ArrayF64
