from maneuvering.orbit.cartesian import Cart
from maneuvering.orbit.convert_kep_cart import convert_cart_to_kep_true, convert_kep_true_to_cart
from maneuvering.orbit.keplerian import KepTrue
from maneuvering.orbit.orbital_system import calc_orb_sys, rot_mat_orb_to_eci
from maneuvering.types import Scalar, Vector3


def apply_impulse_eci(o: KepTrue, imp_eci: Vector3, mu: Scalar) -> KepTrue:
    """
    Применяет мгновенный импульс Δv, заданный в ECI.

    Алгоритм:
    1) Преобразуем орбиту `o` из Кеплеровых (истинных) элементов в декартовы `r, v` (ECI).
    2) Прибавляем импульс скорости в той же СК: `v_new = v + imp_eci`.
    3) Возвращаемся в Кеплеровы (истинные) элементы.

    Parameters
    ----------
    o : KepTrue
        Кеплеровы элементы {a, e, w, i, raan, nu}.
    imp_eci : Vector3
        Импульс скорости Δv в инерциальной системе ECI, [м/с].
    mu : Scalar
        Гравитационный параметр центрального тела, [м^3/с^2].

    Returns
    -------
    KepTrue
        Новое состояние орбиты в виде истинных Кеплеровых элементов.

    Notes
    -----
    - Импульс считается *мгновенным* (позиция `r` не меняется, меняется только `v`).
    """
    cart = convert_kep_true_to_cart(o, mu)
    cart_new = Cart(r=cart.r, v=cart.v + imp_eci)
    return convert_cart_to_kep_true(cart_new, mu)


def apply_impulse_orb(o: KepTrue, imp_orb: Vector3, mu: Scalar) -> KepTrue:
    """
    Применяет мгновенный импульс Δv, заданный в орбитальной СК {r,t,n}, к орбите в ECI.

    Алгоритм:
    1) Преобразуем `o` → декартовы `r, v` в ECI.
    2) Строим орбитальный базис {r̂, t̂, n̂} в текущей точке орбиты по (r, v).
    3) Конвертируем импульс из орбитальной СК в ECI: `imp_eci = R_orb→eci @ imp_orb`.
    4) Применяем импульс: `v_new = v + imp_eci` и конвертируем обратно в истинные элементы.

    Parameters
    ----------
    o : KepTrue
        Кеплеровы элементы {a, e, w, i, raan, nu}.
    imp_orb : Vector3
        Импульс Δv в локальной орбитальной СК {r,t,n} в текущей точке, [м/с].
        Оси: r̂ — по радиусу, t̂ — тангенциальная (в направлении движения), n̂ — вдоль орбитального момента.
    mu : Scalar
        Гравитационный параметр центрального тела, [м^3/с^2].

    Returns
    -------
    KepTrue
        Новое состояние орбиты в виде истинных Кеплеровых элементов.

    Notes
    -----
    - Импульс считается *мгновенным*: `r` неизменен, меняется только `v`.
    - Матрица `R_orb→eci` составлена из столбцов {r̂, t̂, n̂} в координатах ECI.
    """
    cart = convert_kep_true_to_cart(o, mu)

    # Поворот орб.СК→ECI в текущем положении (r,v):
    orb_sys = calc_orb_sys(cart.r, cart.v)
    R = rot_mat_orb_to_eci(orb_sys)
    imp_eci = R @ imp_orb

    cart_new = Cart(r=cart.r, v=cart.v + imp_eci)
    return convert_cart_to_kep_true(cart_new, mu)
