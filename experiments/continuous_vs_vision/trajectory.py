"""Représentations géométriques d'un mot et signatures de chemins de bas niveau."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def word_to_trajectory(
    word: str,
    alphabet: Sequence[str],
    n: int | None = None,
    samples_per_segment: int = 1,
) -> np.ndarray:
    """Convertit un mot en points complexes aux racines n-ièmes de l'unité.

    Les lettres inconnues sont refusées plutôt que supprimées silencieusement. Avec
    `samples_per_segment > 1`, chaque arête est échantillonnée linéairement, sans
    répéter son point de départ.
    """
    letters = list(alphabet)
    n = len(letters) if n is None else n
    if n < 1 or n != len(letters):
        raise ValueError("n doit être égal au nombre de lettres de l'alphabet")
    if len(set(letters)) != len(letters):
        raise ValueError("l'alphabet doit contenir des lettres distinctes")
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment doit être >= 1")
    index = {letter: i for i, letter in enumerate(letters)}
    unknown = [letter for letter in word if letter not in index]
    if unknown:
        raise ValueError(f"lettres absentes de l'alphabet : {unknown!r}")
    knots = np.exp(2j * math.pi * np.arange(n) / n)
    points = np.asarray([knots[index[letter]] for letter in word], dtype=np.complex128)
    if len(points) < 2 or samples_per_segment == 1:
        return points
    sampled = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        sampled.extend(
            start + (end - start) * step / samples_per_segment
            for step in range(1, samples_per_segment + 1)
        )
    return np.asarray(sampled, dtype=np.complex128)


def _xy_points(points: Sequence[complex] | np.ndarray) -> np.ndarray:
    array = np.asarray(points)
    if np.iscomplexobj(array):
        if array.ndim != 1:
            raise ValueError("les points complexes doivent être un vecteur")
        return np.column_stack((array.real, array.imag))
    if array.size == 0:
        return np.zeros((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("les points doivent avoir la forme (m, 2) ou être complexes")
    return array.astype(float, copy=False)


def path_signature(points: Sequence[complex] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les niveaux 1 et 2 de la signature d'un chemin polygonal 2D."""
    xy = _xy_points(points)
    level1 = np.zeros(2, dtype=float)
    level2 = np.zeros((2, 2), dtype=float)
    for delta in np.diff(xy, axis=0):
        level2 += np.outer(level1, delta) + 0.5 * np.outer(delta, delta)
        level1 += delta
    return level1, level2


def chen_product(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Compose des signatures tronquées aux niveaux 1 et 2."""
    left1, left2 = left
    right1, right2 = right
    return (
        left1 + right1,
        left2 + np.outer(left1, right1) + right2,
    )


def trajectory_dft(points: Sequence[complex] | np.ndarray) -> np.ndarray:
    """Transformée de Fourier discrète des échantillons complexes du parcours."""
    return np.fft.fft(np.asarray(points, dtype=np.complex128))
