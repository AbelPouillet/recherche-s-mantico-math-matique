"""Évaluation croisée et conclusion collective.

Un modèle est écarté du consensus s'il dépasse le budget (« bavard »), si son coût
déclaré s'écarte du coût mesuré de plus de `tolerance_pct` (« optimiste »), ou si sa
sortie est invalide. Aucun aléa : le résultat dépend uniquement des entrées.
"""
from __future__ import annotations

from .adapters import CORRELATION_KEYS, measured_cost


def budget_gap_pct(measured: int, declared: int) -> float:
    return round((measured - declared) / declared * 100.0, 2)


def assess(parsed: dict | None, errors: list[str], budget: int, tolerance_pct: float) -> dict:
    if parsed is None:
        return {"valid": False, "errors": errors, "flags": ["invalide"], "included": False}
    measured = measured_cost(parsed["claims"])
    gap = budget_gap_pct(measured, parsed["declared_cost"])
    flags = []
    if measured > budget:
        flags.append("bavard")
    if abs(gap) > tolerance_pct:
        flags.append("ecart_budget")
    return {"valid": True, "errors": [], "measured_cost": measured,
            "declared_cost": parsed["declared_cost"], "budget_gap_pct": gap,
            "flags": flags, "included": not flags}


def cross_evaluate(responses: dict[str, dict]) -> dict:
    """Accord par paire = 1 - écart absolu moyen des corrélations (sur les réponses valides)."""
    names = sorted(responses)
    matrix: dict[str, dict[str, float]] = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            diffs = [abs(responses[a]["correlations"][k] - responses[b]["correlations"][k])
                     for k in CORRELATION_KEYS]
            matrix[a][b] = round(1.0 - sum(diffs) / len(diffs), 3)
    return matrix


def collective(responses: dict[str, dict], assessments: dict[str, dict], weights: dict[str, float],
               h0_threshold: float) -> dict:
    kept = sorted(n for n, a in assessments.items() if a["included"])
    excluded = {n: assessments[n]["flags"] for n in sorted(assessments) if not assessments[n]["included"]}
    if not kept:
        return {"kept": [], "excluded": excluded, "correlations": None, "verdict": "aucun consensus"}
    mean = {k: round(sum(responses[n]["correlations"][k] for n in kept) / len(kept), 3)
            for k in CORRELATION_KEYS}
    score = round(sum(weights[k] * mean[k] for k in CORRELATION_KEYS) / sum(weights.values()), 3)
    spread = {k: round(max(responses[n]["correlations"][k] for n in kept)
                       - min(responses[n]["correlations"][k] for n in kept), 3)
              for k in CORRELATION_KEYS}
    return {"kept": kept, "excluded": excluded, "correlations": mean, "spread": spread,
            "weighted_score": score,
            "verdict": "H0 retenue" if score < h0_threshold else "à tester contre baselines"}
