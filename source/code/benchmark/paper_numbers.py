"""Rounding shared by the manuscript macros and the paper tables.

Cell rates over 400 runs and regression contrasts end in 5 at the second
decimal (63.75, 24.75, 0.25).  Floating-point noise in the analysis output
turns some of them into 63.749999 and others into 0.250001, so Python's
default formatting printed 63.7 next to 0.3.  This helper first removes
the noise and then rounds half away from zero, so every value that ends
in 5 rounds the same way in the text and in the tables.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def round_half_up(value: float, decimals: int = 1) -> Decimal:
    """Round ``value`` to ``decimals`` places, half away from zero, after removing float noise."""
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(repr(round(float(value), 8))).quantize(quantum, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded == 0 else rounded


def fixed(value: float, decimals: int = 1, thousands: bool = False) -> str:
    """Format ``value`` with ``decimals`` places and half-up rounding."""
    return f'{round_half_up(value, decimals):{"," if thousands else ""}f}'
