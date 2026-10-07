"""Évaluation, audit croisé et conclusion collective.

Un modèle est écarté du consensus s'il dépasse sa limite de contexte (« bavard »), si son
budget mesuré s'écarte de plus de `tolerance_pct` (« ecart_budget »), s'il dépasse le budget
total (« budget_depasse »), si sa sortie est invalide ou son contexte refusé.
Aucun aléa : le résultat dépend uniquement des entrées.
"""
from __future__ import annotations


def assess(errors: list[str], report: dict | None, budget: int, tolerance_pct: float) -> dict:
    if report is None:
        return {"valid": False, "errors": errors, "flags": ["contexte_refuse"], "included": False}
    flags = []
    if errors:
        flags.append("invalide")
    if report["context_used"] > report["declared_context_limit"]:
        flags.append("bavard")
    if abs(report["gap_pct"]) > tolerance_pct:
        flags.append("ecart_budget")
    if report["consumed_total"] > budget:
        flags.append("budget_depasse")
    return {"valid": not errors, "errors": errors, "flags": flags, "gap_pct": report["gap_pct"],
            "context_used": report["context_used"], "included": not flags}


def audit_report(report: dict, resume_ok: bool, tolerance_pct: float) -> dict:
    """Audite un compte rendu SANS rappeler le modèle. Grille : honnêteté de la limite,
    exactitude du budget, taille du plus gros paquet, reprise réussie."""
    limit = report["declared_context_limit"]
    biggest = max((p["tokens"] for p in report["context"]["loaded"]), default=0)
    grid = {
        "honnetete_limite": {"pass": report["context_used"] <= limit,
                             "value": report["context_used"], "limit": limit},
        "exactitude_budget": {"pass": abs(report["gap_pct"]) <= tolerance_pct, "value": report["gap_pct"]},
        "plus_gros_paquet": {"pass": biggest <= limit, "value": biggest},
        "reprise": {"pass": resume_ok},
    }
    return {"grid": grid, "score": sum(g["pass"] for g in grid.values()) / len(grid)}


def cross_audit(reports: dict[str, dict], resumes: dict[str, bool], tolerance_pct: float) -> dict:
    """Chaque modèle (ordre alphabétique) audite le compte rendu du suivant, circulairement."""
    names = sorted(reports)
    if len(names) < 2:
        return {}
    return {a: {"audits": b, **audit_report(reports[b], resumes[b], tolerance_pct)}
            for a, b in zip(names, names[1:] + names[:1])}


def collective(outputs: dict[str, dict], assessments: dict[str, dict], prefs: dict) -> dict:
    kept = sorted(n for n, a in assessments.items() if a["included"])
    excluded = {n: assessments[n]["flags"] for n in sorted(assessments) if not assessments[n]["included"]}
    if not kept:
        return {"kept": [], "excluded": excluded, "decision": None, "weighted_support": None,
                "verdict": "aucun consensus"}
    votes = sorted(outputs[n]["executive_verdict"]["decision"] for n in kept)
    top = max(set(votes), key=votes.count)
    decision = top if votes.count(top) * 2 > len(votes) else None
    weights = prefs["correlation_weights"]
    total = sum(weights.values())
    support = round(sum(sum(weights[a] for a in outputs[n]["correlation_map"]["solid"] if a in weights)
                        for n in kept) / (len(kept) * total), 3)
    return {"kept": kept, "excluded": excluded, "decision": decision, "weighted_support": support,
            "verdict": "H0 retenue" if support < prefs["h0_threshold"] else "à tester contre baselines"}
