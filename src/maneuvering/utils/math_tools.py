def clamp(x: float, lo: float, hi: float) -> float:
    """Ограничение значения x в диапазоне [lo, hi]."""
    return max(lo, min(hi, x))
