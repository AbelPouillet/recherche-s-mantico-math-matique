#!/usr/bin/env python3
"""Banc reproductible de prédiction du caractère suivant (régression ridge numpy)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedbabel.gematria import get_alphabet
from experiments.continuous_vs_vision.baselines import (
    one_hot,
    random_circle_permutation,
    rasterize,
)

MODES = ("one_hot", "circle", "permuted_circle", "raster", "random_pattern")


def load_words(
    lang: str, data_dir: Path, max_words: int, seed: int
) -> tuple[list[str], str]:
    """Charge les entrées d'un TSV local ou retourne un corpus jouet déclaré."""
    alphabet = get_alphabet(lang)
    path = data_dir / f"{lang}.tsv"
    words: set[str] = set()
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                word = "".join(alphabet.letters_of(row.get("word", "")))
                if len(word) >= 2:
                    words.add(word)
        source = f"dictionnaire local: {path}"
    else:
        toy = (
            "bonjour", "bonsoir", "maison", "raison", "saison", "parler",
            "manger", "chanter", "livre", "arbre", "fleur", "route",
            "soleil", "lune", "langue", "lettre", "cercle", "chemin",
            "riviere", "montagne", "petit", "grand", "rouge", "bleu",
            "voiture", "fenetre", "musique", "histoire", "ami", "amie",
        )
        words = {"".join(alphabet.letters_of(word)) for word in toy}
        words = {word for word in words if len(word) >= 2}
        source = "mini-corpus jouet intégré (aucun dictionnaire local trouvé)"
    ordered = sorted(words)
    if len(ordered) < 2:
        raise ValueError(f"corpus insuffisant pour séparer entraînement/test : {source}")
    if len(ordered) > max_words:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(ordered), size=max_words, replace=False)
        ordered = sorted(ordered[int(i)] for i in chosen)
    return ordered, source


def _examples(
    words: Sequence[str], context_size: int, alphabet: Sequence[str]
) -> tuple[list[list[str]], np.ndarray]:
    index = {letter: i for i, letter in enumerate(alphabet)}
    contexts: list[list[str]] = []
    labels: list[int] = []
    for word in words:
        chars = list(word)
        for target_index, target in enumerate(chars):
            contexts.append(chars[max(0, target_index - context_size):target_index])
            labels.append(index[target])
    return contexts, np.asarray(labels, dtype=int)


def _encode(
    context: Sequence[str],
    mode: str,
    alphabet: Sequence[str],
    context_size: int,
    feature_size: int,
    raster_size: int,
    permutation: dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    vocab_size = len(alphabet)
    if mode == "one_hot":
        raw = one_hot(context, alphabet).reshape(-1)
        raw = np.pad(raw, (max(0, context_size * vocab_size - len(raw)), 0))
    elif mode in ("circle", "permuted_circle"):
        positions = {letter: i for i, letter in enumerate(alphabet)}
        if mode == "permuted_circle":
            positions = permutation
        raw = np.zeros(context_size * 2, dtype=np.float32)
        offset = context_size - len(context)
        for i, letter in enumerate(context, offset):
            angle = 2 * math.pi * positions[letter] / vocab_size
            raw[2 * i:2 * i + 2] = (math.cos(angle), math.sin(angle))
    elif mode == "raster":
        raw = rasterize(context, alphabet, size=raster_size).reshape(-1)
    elif mode == "random_pattern":
        random_units = [alphabet[int(i)] for i in rng.integers(vocab_size, size=len(context))]
        raw = one_hot(random_units, alphabet).reshape(-1)
        raw = np.pad(raw, (max(0, context_size * vocab_size - len(raw)), 0))
    else:
        raise ValueError(f"mode inconnu : {mode}")
    if len(raw) > feature_size:
        raise ValueError("feature_size insuffisant pour cet encodage")
    return np.pad(np.asarray(raw, dtype=np.float64), (0, feature_size - len(raw)))


def _fit_ridge(features: np.ndarray, labels: np.ndarray, classes: int) -> np.ndarray:
    """Régression ridge multi-sortie; tous les modes partagent la même taille."""
    design = np.column_stack((features, np.ones(len(features))))
    targets = np.eye(classes, dtype=np.float64)[labels]
    regularizer = np.eye(design.shape[1], dtype=np.float64) * 1e-2
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ targets)


def _permutation_pvalue(differences: np.ndarray, count: int, seed: int) -> float:
    """Test bilatéral apparié par permutation de signes avec correction add-one."""
    if not len(differences):
        return 1.0
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(count):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        exceed += abs(float(np.mean(differences * signs))) >= observed
    return (exceed + 1) / (count + 1)


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Correction de Holm (FWER) sur la famille des comparaisons fournie."""
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def run(
    lang: str = "fr",
    seed: int = 17,
    max_words: int = 1000,
    permutations: int = 1000,
    raster_size: int = 16,
    data_dir: Path | None = None,
) -> dict:
    if max_words < 2 or permutations < 1 or raster_size < 1:
        raise ValueError("max_words >= 2, permutations >= 1 et raster_size >= 1 requis")
    alphabet = get_alphabet(lang).letters
    words, source = load_words(lang, data_dir or ROOT / "data" / "dict", max_words, seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(words))
    split = max(1, int(0.8 * len(words)))
    train_words = [words[int(i)] for i in order[:split]]
    test_words = [words[int(i)] for i in order[split:]]
    if not test_words:
        test_words = train_words[-1:]
        train_words = train_words[:-1]
    train_contexts, train_labels = _examples(train_words, 4, alphabet)
    test_contexts, test_labels = _examples(test_words, 4, alphabet)

    feature_size = max(len(alphabet) * 4, raster_size * raster_size)
    permuted = random_circle_permutation(alphabet, seed)
    train_features: dict[str, np.ndarray] = {}
    test_features: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for mode_index, mode in enumerate(MODES):
        mode_rng = np.random.default_rng(seed + mode_index + 1)
        train_features[mode] = np.vstack([
            _encode(ctx, mode, alphabet, 4, feature_size, raster_size, permuted, mode_rng)
            for ctx in train_contexts
        ])
        test_features[mode] = np.vstack([
            _encode(ctx, mode, alphabet, 4, feature_size, raster_size, permuted, mode_rng)
            for ctx in test_contexts
        ])
        weights = _fit_ridge(train_features[mode], train_labels, len(alphabet))
        design = np.column_stack((test_features[mode], np.ones(len(test_features[mode]))))
        predictions[mode] = (design @ weights).argmax(axis=1)

    correctness = {
        mode: (predictions[mode] == test_labels).astype(float) for mode in MODES
    }
    raw_p = {
        mode: _permutation_pvalue(
            correctness[mode] - correctness["one_hot"], permutations, seed + i + 50
        )
        for i, mode in enumerate(MODES)
        if mode != "one_hot"
    }
    adjusted = _holm(raw_p)
    results = {
        mode: {
            "accuracy": float(np.mean(correctness[mode])),
            "parameter_count": int((feature_size + 1) * len(alphabet)),
            "inference_flops_per_example": int(
                2 * (feature_size + 1) * len(alphabet)
            ),
            "permutation_p_vs_one_hot": raw_p.get(mode),
            "holm_p_vs_one_hot": adjusted.get(mode),
        }
        for mode in MODES
    }
    return {
        "task": "prédiction du caractère suivant",
        "language": lang,
        "corpus_source": source,
        "toy_corpus": source.startswith("mini-corpus jouet"),
        "seed": seed,
        "word_count": len(words),
        "train_word_count": len(train_words),
        "test_word_count": len(test_words),
        "test_character_count": int(len(test_labels)),
        "model": "régression ridge multi-sortie numpy",
        "parameter_matching": "nombre de paramètres identique entre représentations",
        "feature_size": feature_size,
        "permutation_test_count": permutations,
        "multiple_comparison_correction": "Holm",
        "metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-words", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--raster-size", type=int, default=16)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "dict")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        lang=args.lang,
        seed=args.seed,
        max_words=args.max_words,
        permutations=args.permutations,
        raster_size=args.raster_size,
        data_dir=args.data_dir,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
