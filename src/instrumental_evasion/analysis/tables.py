"""Plain-text table formatting shared by the analysis commands."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def one_decimal(value: float | None) -> str:
    """Round half up to one decimal, as in the paper's tables."""
    if value is None:
        return "-"
    return str(Decimal(repr(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def percent(numerator: int, denominator: int) -> str:
    """Return ``k/n (p%)``, or ``k/n`` when the denominator is zero."""
    if not denominator:
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator} ({one_decimal(100 * numerator / denominator)}%)"


def text_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Left-align the first column and right-align the others."""
    cells = [[str(cell) for cell in row] for row in (headers, *rows)]
    widths = [max(len(cell) for cell in column) for column in zip(*cells)]
    return "\n".join(
        "  ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        ).rstrip()
        for row in cells
    )
