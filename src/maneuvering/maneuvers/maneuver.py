from dataclasses import dataclass

from maneuvering.types import Scalar, Vector3


@dataclass(frozen=True, slots=True)
class Maneuver:
    """
    Импульсный манёвр

    Атрибуты
    --------
    dv : Vector3
        Импульс скорости, [м/с].
    angle : Scalar
         Угол приложения манёвра относительно некоторой точки, [рад].
    """
    dv: Vector3
    angle: Scalar


@dataclass(frozen=True, slots=True)
class ManeuverT:
    """
    Импульсный манёвр

    Атрибуты
    --------
    dv : Vector3
        Импульс скорости, [м/с].
    t : Scalar
         Время приложения манёвра относительно некоторого момента, [сек].
    """
    dv: Vector3
    t: Scalar
