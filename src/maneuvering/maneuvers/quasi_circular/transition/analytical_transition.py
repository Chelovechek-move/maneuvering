from __future__ import annotations

import numpy as np
from typing import List

from maneuvering.maneuvers.maneuver import Maneuver
from maneuvering.maneuvers.quasi_circular.transition.coplanar.coplanar import coplanar_analytical
from maneuvering.maneuvers.quasi_circular.transition.noncoplanar.noncoplanar import noncoplanar_analytical
from maneuvering.maneuvers.quasi_circular.reference_orbit import trans_devs
from maneuvering.orbit.keplerian import Kep
from maneuvering.types import Scalar


def analytical_transition(oi: Kep, ot: Kep, mu: Scalar, tol: Scalar = 2.220446049250313e-14) -> List[Maneuver]:
    devs = trans_devs(oi, ot)
    return coplanar_analytical(oi, ot, mu) if abs(devs.i) < tol else noncoplanar_analytical(oi, ot, mu)
