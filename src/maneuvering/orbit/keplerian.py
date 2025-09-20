from dataclasses import dataclass

from maneuvering.types import Scalar


@dataclass(frozen=True, slots=True)
class Kep:
    """
    Кеплеровы элементы орбиты {a, e, w, i, raan}.

    Атрибуты
    --------
    a : Scalar
        Большая полуось, [м].
    e : Scalar
        Эксцентриситет, [1].
    w : Scalar
        Аргумент перицентра, [рад].
    i : Scalar
        Наклонение, [рад].
    raan : Scalar
        Долгота восходящего узла, [рад].
    """
    a: Scalar
    e: Scalar
    w: Scalar
    i: Scalar
    raan: Scalar


@dataclass(frozen=True, slots=True)
class KepTrue:
    """
    Истинные Кеплеровы элементы {a, e, w, i, raan, nu}.

    Атрибуты
    --------
    orb : Kep
        Кеплеровы элементы орбиты.
    nu : Scalar
        Истинная аномалия, [рад].
    """
    orb: Kep
    nu: Scalar


@dataclass(frozen=True, slots=True)
class KepMean:
    """
    Средние Кеплеровы элементы {a, e, w, i, raan, M}.

    Атрибуты
    --------
    orb : Kep
        Кеплеровы элементы орбиты.
    M : Scalar
        Средняя аномалия, [рад].
    """
    orb: Kep
    M: Scalar


@dataclass(frozen=True, slots=True)
class KepEcc:
    """
    Эксцентрические Кеплеровы элементы {a, e, w, i, raan, Е}.

    Атрибуты
    --------
    orb : Kep
        Кеплеровы элементы орбиты.
    E : Scalar
        Эксцентрическая аномалия, [рад].
    """
    orb: Kep
    E: Scalar


def keplerian(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar) -> Kep:
    """Фабрика для базовых Кеплеровых элементов."""
    return Kep(a=a, e=e, w=w, i=i, raan=raan)


def kep_true(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar, nu: Scalar) -> KepTrue:
    """Фабрика для истинных Кеплеровых элементов"""
    return KepTrue(orb=Kep(a=a, e=e, w=w, i=i, raan=raan), nu=nu)


def kep_mean(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar, M: Scalar) -> KepMean:
    """Фабрика для средних Кеплеровых элементов"""
    return KepMean(orb=Kep(a=a, e=e, w=w, i=i, raan=raan), M=M)


def kep_ecc(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar, E: Scalar) -> KepEcc:
    """Фабрика для эксцентрических Кеплеровых элементов"""
    return KepEcc(orb=Kep(a=a, e=e, w=w, i=i, raan=raan), E=E)
