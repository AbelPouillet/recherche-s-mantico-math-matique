"""Préfixes et sous-chaînes contiguës d'une suite d'unités (lettres/caractères)."""
from __future__ import annotations

from typing import Iterator, Sequence


def prefixes(units: Sequence[str]) -> list[str]:
    """Tous les préfixes non vides, du plus court au mot entier."""
    return ["".join(units[:i]) for i in range(1, len(units) + 1)]


def contiguous_substrings(units: Sequence[str], max_len: int | None = None
                          ) -> Iterator[tuple[int, str]]:
    """(position de début, sous-chaîne) pour toutes les sous-chaînes contiguës,
    avec longueur <= max_len si fourni."""
    n = len(units)
    for start in range(n):
        stop_max = n if max_len is None else min(n, start + max_len)
        for stop in range(start + 1, stop_max + 1):
            yield start, "".join(units[start:stop])


def count_substring_occurrences(n: int, max_len: int | None = None) -> int:
    """Nombre d'occurrences de sous-chaînes (avec répétitions) d'un mot de n unités."""
    if max_len is None or max_len >= n:
        return n * (n + 1) // 2
    return sum(n - k + 1 for k in range(1, max_len + 1))
