"""Validation de la sortie JSON stricte de EMBEDBABEL_BENCH_V2.md.

Ce que V2 impose (lu dans bench/prompts/EMBEDBABEL_BENCH_V2.md) :
  - JSON uniquement, aucun texte hors JSON ;
  - 10 clés de premier niveau ; 6 tags ; 3 à 7 intuitions avec SIF ; un SDM ;
  - invalide si : champ manquant, tag hors vocabulaire, aucun test de réfutation,
    aucune baseline, SDM/SIF sans justification.
V2 ne définit PAS les sous-champs. Ceux ci-dessous sont une convention du harnais
(schéma « v2-conv-0.2 »), documentée dans bench/harnesses/embedbabel-bench/0.2.0/HARNESS.md.
"""
from __future__ import annotations

import json

TAGS = ("DEFINI", "TESTABLE", "PLAUSIBLE", "SPECULATIF", "PROBABLEMENT_FAUX", "NON_FALSIFIABLE")
TOP_KEYS = ("meta", "executive_verdict", "formalization", "correlation_map", "anti_mystical_audit",
            "fruitful_intuitions", "experimental_plan", "system_integration",
            "smallest_decisive_experiment", "claim_tagging_summary")
DECISIONS = ("GO", "NO-GO")
CORRELATION_CLASSES = ("solid", "fragile", "illusory")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _text(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def validate(text: str) -> tuple[dict | None, list[str]]:
    """Retourne (objet, erreurs). L'objet est None dès qu'il y a une erreur."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, [f"texte hors JSON ou JSON invalide : {exc}"]
    if not isinstance(obj, dict):
        return None, ["la racine doit être un objet JSON"]
    errors = [f"champ manquant : {k}" for k in TOP_KEYS if k not in obj]
    if errors:
        return None, errors

    meta = obj["meta"]
    if not isinstance(meta, dict) or not _text(meta.get("model")):
        errors.append("meta.model manquant")
    ev = obj["executive_verdict"]
    if not isinstance(ev, dict) or ev.get("decision") not in DECISIONS:
        errors.append(f"executive_verdict.decision doit être dans {DECISIONS}")
    cm = obj["correlation_map"]
    if not isinstance(cm, dict) or any(not isinstance(cm.get(c), list) for c in CORRELATION_CLASSES):
        errors.append(f"correlation_map doit avoir les listes {CORRELATION_CLASSES}")

    audit = obj["anti_mystical_audit"]
    if not isinstance(audit, dict) or not _num(audit.get("sdm")) or not 0 <= audit["sdm"] <= 100:
        errors.append("anti_mystical_audit.sdm doit être un nombre entre 0 et 100")
    elif not _text(audit.get("justification")):
        errors.append("SDM sans justification")

    intu = obj["fruitful_intuitions"]
    if not isinstance(intu, list) or not 3 <= len(intu) <= 7:
        errors.append("fruitful_intuitions doit contenir de 3 à 7 intuitions")
    else:
        for i, it in enumerate(intu):
            if not isinstance(it, dict) or not _num(it.get("sif")) or not 0 <= it["sif"] <= 100:
                errors.append(f"intuition {i} : sif doit être un nombre entre 0 et 100")
            elif not _text(it.get("justification")):
                errors.append(f"SIF sans justification (intuition {i})")

    plan = obj["experimental_plan"]
    if not isinstance(plan, dict) or not isinstance(plan.get("baselines"), list) or not plan["baselines"]:
        errors.append("aucune baseline")

    summary = obj["claim_tagging_summary"]
    claims = summary.get("claims") if isinstance(summary, dict) else None
    if not isinstance(claims, list) or not claims:
        errors.append("claim_tagging_summary.claims doit être une liste non vide")
    else:
        refutations = 0
        for i, c in enumerate(claims):
            if not isinstance(c, dict) or not _text(c.get("text")):
                errors.append(f"claim {i} : texte manquant")
                continue
            if c.get("tag") not in TAGS:
                errors.append(f"tag hors vocabulaire : {c.get('tag')!r} (claim {i})")
            if _text(c.get("refutation_test")):
                refutations += 1
            elif not _text(c.get("justification")):
                errors.append(f"claim {i} sans justification ni test de réfutation")
        if refutations == 0:
            errors.append("aucun test de réfutation")

    for key in ("formalization", "system_integration", "smallest_decisive_experiment"):
        if not obj[key]:
            errors.append(f"{key} vide")
    return (None, errors) if errors else (obj, [])
