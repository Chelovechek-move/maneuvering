from dataclasses import dataclass

from maneuvering.types import Vector3


@dataclass(frozen=True, slots=True)
class Cart:
    """
    Декартовы элементы орбиты {r, v}, r={r_1, r_2, r_3}, v={v_1, v_2, v_3}.

    Атрибуты
    --------
    r : Vector3
        Радиус-вектор, [м].
    v : Vector3
        Вектор скорости, [м/с].
    """

    r: Vector3
    v: Vector3
