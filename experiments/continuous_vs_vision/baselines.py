"""Encodages de référence pour le banc flux continu contre vision."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def one_hot(units: Sequence[str], alphabet: Sequence[str]) -> np.ndarray:
    """Encode une séquence en matrice one-hot (temps, alphabet)."""
    letters = list(alphabet)
    if len(set(letters)) != len(letters):
        raise ValueError("l'alphabet doit contenir des lettres distinctes")
    index = {letter: i for i, letter in enumerate(letters)}
    result = np.zeros((len(units), len(letters)), dtype=np.float32)
    for row, unit in enumerate(units):
        if unit not in index:
            raise ValueError(f"unité absente de l'alphabet : {unit!r}")
        result[row, index[unit]] = 1.0
    return result


def random_circle_permutation(
    alphabet: Sequence[str], seed: int
) -> dict[str, int]:
    """Associe chaque lettre à une position circulaire permutée et reproductible."""
    letters = list(alphabet)
    if len(set(letters)) != len(letters):
        raise ValueError("l'alphabet doit contenir des lettres distinctes")
    positions = np.random.default_rng(seed).permutation(len(letters))
    return dict(zip(letters, map(int, positions)))


def rasterize(
    units: Sequence[str], alphabet: Sequence[str], size: int = 32
) -> np.ndarray:
    """Rasterise via l'implémentation canonique `viz.circle_trace.trace_to_array`."""
    if size < 1:
        raise ValueError("size doit être >= 1")
    from viz.circle_trace import trace_to_array

    return trace_to_array(units, alphabet, size=size).image
