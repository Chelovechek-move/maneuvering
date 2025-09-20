from dataclasses import dataclass

from maneuvering.types import Scalar


@dataclass(frozen=True, slots=True)
class Keplerian:
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
class KeplerianTrue:
    """
    Истинные Кеплеровы элементы {a, e, w, i, raan, nu}.

    Атрибуты
    --------
    orb : Keplerian
        Кеплеровы элементы орбиты.
    nu : Scalar
        Истинная аномалия, [рад].
    """
    orb: Keplerian
    nu: Scalar


@dataclass(frozen=True, slots=True)
class KeplerianMean:
    """
    Средние Кеплеровы элементы {a, e, w, i, raan, M}.

    Атрибуты
    --------
    orb : Keplerian
        Кеплеровы элементы орбиты.
    M : Scalar
        Средняя аномалия, [рад].
    """
    orb: Keplerian
    M: Scalar


@dataclass(frozen=True, slots=True)
class KeplerianEcc:
    """
    Эксцентрические Кеплеровы элементы {a, e, w, i, raan, Е}.

    Атрибуты
    --------
    orb : Keplerian
        Кеплеровы элементы орбиты.
    E : Scalar
        Эксцентрическая аномалия, [рад].
    """
    orb: Keplerian
    E: Scalar


def keplerian(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar) -> Keplerian:
    """Фабрика для базовых Кеплеровых элементов."""
    return Keplerian(a=a, e=e, w=w, i=i, raan=raan)


def kep_true(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar, nu: Scalar) -> KeplerianTrue:
    """Фабрика для истинных Кеплеровых элементов"""
    return KeplerianTrue(orb=Keplerian(a=a, e=e, w=w, i=i, raan=raan), nu=nu)


def kep_mean(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar, M: Scalar) -> KeplerianMean:
    """Фабрика для средних Кеплеровых элементов"""
    return KeplerianMean(orb=Keplerian(a=a, e=e, w=w, i=i, raan=raan), M=M)


def kep_ecc(a: Scalar, e: Scalar, w: Scalar, i: Scalar, raan: Scalar, E: Scalar) -> KeplerianEcc:
    """Фабрика для эксцентрических Кеплеровых элементов"""
    return KeplerianEcc(orb=Keplerian(a=a, e=e, w=w, i=i, raan=raan), E=E)
