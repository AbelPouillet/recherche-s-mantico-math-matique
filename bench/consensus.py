"""Évaluation, grille de comptes rendus et conclusion collective.

Ce que ce module décide
----------------------
Un modèle est écarté du consensus s'il dépasse sa limite de contexte (`bavard`), s'il dépasse le
budget total (`budget_depasse`), si sa sortie est invalide (`invalide`) ou si son contexte a été
refusé (`contexte_refuse`).

`ecart_budget` ne s'applique **qu'aux adaptateurs qui annoncent un budget** — les factices, dont
c'est la raison d'être (`optimiste` annonce trois fois trop bas). Un adaptateur réel n'annonce rien :
le harnais lui impose une répartition neutre `budget / 3`, donc l'écart mesuré ne dit rien de
l'honnêteté du modèle. Appliqué aux modèles réels, ce drapeau les excluait tous : trois campagnes
réelles s'étaient terminées en « aucun consensus » pour une raison arithmétique. L'écart reste
publié comme **métrique** (`budget_gap_pct`), il ne sert plus à exclure.

Ce que la conclusion collective n'est pas
-----------------------------------------
Elle agrège des **opinions de modèles** : `auto_declaration_agregee` est la moyenne pondérée des
axes que chaque modèle inclus a lui-même classés `solid`. C'est une auto-évaluation, pas une mesure.
Aucun lecteur de `report.json` ne doit pouvoir confondre les deux : le champ est nommé pour ce qu'il
est, et `decision_kind` le dit explicitement.
"""
from __future__ import annotations

DECISION_KIND = "opinion_agregee_de_modeles"


def _budget_used(report: dict) -> int:
    """Tokens réellement consommés en réponse, compteur du serveur si disponible, sinon `chars/4`."""
    real = report.get("consumed_total_real")
    return real if isinstance(real, int) and real > 0 else report["consumed_total"]


def assess(errors: list[str], report: dict | None, budget: int, tolerance_pct: float,
           declares_budget: bool = True) -> dict:
    """Drapeaux d'un modèle. `declares_budget=False` : l'écart de budget n'est plus un motif d'exclusion."""
    if report is None:
        return {"valid": False, "errors": errors, "flags": ["contexte_refuse"], "included": False}
    flags = []
    if errors:
        flags.append("invalide")
    if report["context_used"] > report["declared_context_limit"]:
        flags.append("bavard")
    if declares_budget and abs(report["gap_pct"]) > tolerance_pct:
        flags.append("ecart_budget")
    used = _budget_used(report)
    if used > budget:
        flags.append("budget_depasse")
    return {"valid": not errors, "errors": errors, "flags": flags, "gap_pct": report["gap_pct"],
            "budget_gap_pct": report["gap_pct"], "budget_used": used,
            "budget_used_source": "serveur" if report.get("consumed_total_real") else "chars/4",
            "declares_budget": declares_budget,
            "context_used": report["context_used"], "included": not flags}


def grille_compte_rendu(report: dict, resume_ok: bool, tolerance_pct: float,
                        declares_budget: bool = True, budget: int | None = None) -> dict:
    """Grille d'un compte rendu, **sans rappeler le modèle** et sans qu'aucun modèle n'intervienne.

    Quatre critères. Le deuxième porte sur ce qui est réellement vérifiable : l'écart au budget
    annoncé quand le modèle en annonce un, le respect du budget sinon.
    """
    limit = report["declared_context_limit"]
    biggest = max((p["tokens"] for p in report["context"]["loaded"]), default=0)
    used = _budget_used(report)
    budget = budget if budget is not None else report.get("budget")
    grid = {
        "honnetete_limite": {"pass": report["context_used"] <= limit,
                             "value": report["context_used"], "limit": limit},
        "plus_gros_paquet": {"pass": biggest <= limit, "value": biggest},
        "reprise": {"pass": resume_ok},
    }
    if declares_budget:
        grid["exactitude_budget"] = {"pass": abs(report["gap_pct"]) <= tolerance_pct,
                                     "value": report["gap_pct"], "tolerance_pct": tolerance_pct}
    else:
        grid["budget_respecte"] = {"pass": budget is not None and 0 < used <= budget,
                                   "value": used, "budget": budget}
    return {"grid": grid, "score": sum(g["pass"] for g in grid.values()) / len(grid),
            "declares_budget": declares_budget}


def grille_comptes_rendus(reports: dict[str, dict], resumes: dict[str, bool], tolerance_pct: float,
                          declares_budget: dict[str, bool] | None = None,
                          budget: int | None = None) -> dict:
    """Grille déterministe de chaque compte rendu, dans l'ordre alphabétique.

    Ancien nom : `cross_audit`. Aucun modèle n'audite quoi que ce soit ici : la grille est calculée
    par le harnais. Le nom précédent laissait croire que le modèle `a` auditait le compte rendu de
    `b`, ce qui était faux et se lisait pourtant tel quel dans `report.json`.
    """
    flags = declares_budget or {}
    return {name: grille_compte_rendu(report, resumes.get(name, False), tolerance_pct,
                                      flags.get(name, True), budget)
            for name, report in sorted(reports.items())}


def collective(outputs: dict[str, dict], assessments: dict[str, dict], prefs: dict) -> dict:
    """Agrégat **d'opinions** des modèles retenus. Voir l'avertissement en tête de module."""
    kept = sorted(n for n, a in assessments.items() if a["included"])
    excluded = {n: assessments[n]["flags"] for n in sorted(assessments) if not assessments[n]["included"]}
    if not kept:
        return {"kept": [], "excluded": excluded, "decision": None, "decision_kind": DECISION_KIND,
                "votes": {}, "auto_declaration_agregee": None,
                "auto_declaration_interpretation": "aucun modèle retenu : rien à agréger",
                "verdict": "aucun modèle retenu"}
    votes = sorted(outputs[n]["executive_verdict"]["decision"] for n in kept)
    top = max(set(votes), key=votes.count)
    decision = top if votes.count(top) * 2 > len(votes) else None
    weights = prefs["correlation_weights"]
    total = sum(weights.values())
    support = round(sum(sum(weights[a] for a in outputs[n]["correlation_map"]["solid"] if a in weights)
                        for n in kept) / (len(kept) * total), 3)
    if decision is None:
        verdict = "aucune décision majoritaire"
    else:
        verdict = f"opinion agrégée : {decision}"
    return {
        "kept": kept, "excluded": excluded, "decision": decision, "decision_kind": DECISION_KIND,
        "votes": {d: votes.count(d) for d in sorted(set(votes))},
        # ce que les modèles ont dit d'eux-mêmes — à ne jamais présenter comme un résultat
        "auto_declaration_agregee": support,
        "auto_declaration_seuil": prefs["h0_threshold"],
        "auto_declaration_sous_seuil": support < prefs["h0_threshold"],
        "auto_declaration_interpretation": (
            "moyenne pondérée des axes que les modèles inclus ont eux-mêmes classés « solid » ; "
            "auto-évaluation, pas une mesure"),
        "verdict": verdict,
    }
