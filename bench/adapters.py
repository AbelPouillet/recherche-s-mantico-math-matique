"""Adaptateurs factices (sans réseau) + validation stricte de la sortie JSON.

Chaque adaptateur renvoie une chaîne JSON. Le harnais mesure lui-même le coût
(somme des longueurs des affirmations) et le compare au coût déclaré.
"""
from __future__ import annotations

import json
import random

TAGS = ("DEFINI", "TESTABLE", "PLAUSIBLE", "SPECULATIF", "PROBABLEMENT_FAUX")
CORRELATION_KEYS = ("word~etymon", "etymon~phoneme", "word~phoneme")


def measured_cost(claims: list[dict]) -> int:
    return sum(len(str(c.get("text", ""))) for c in claims)


def validate_response(text: str) -> tuple[dict | None, list[str]]:
    """Retourne (objet, erreurs). Objet = None si le JSON est invalide ou hors schéma."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, [f"JSON invalide : {exc}"]
    errors: list[str] = []
    if not isinstance(obj, dict):
        return None, ["la racine doit être un objet"]
    for key in ("model", "claims", "declared_cost", "correlations", "verdict"):
        if key not in obj:
            errors.append(f"clé manquante : {key}")
    if errors:
        return None, errors
    if not isinstance(obj["claims"], list) or not obj["claims"]:
        errors.append("claims doit être une liste non vide")
    else:
        for i, c in enumerate(obj["claims"]):
            if not isinstance(c, dict) or not isinstance(c.get("text"), str) or c.get("tag") not in TAGS:
                errors.append(f"claims[{i}] : texte ou tag invalide")
    if not isinstance(obj["declared_cost"], int) or obj["declared_cost"] <= 0:
        errors.append("declared_cost doit être un entier > 0")
    corr = obj["correlations"]
    if not isinstance(corr, dict) or set(corr) != set(CORRELATION_KEYS):
        errors.append(f"correlations doit avoir exactement les clés {CORRELATION_KEYS}")
    else:
        for k, v in corr.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= v <= 1.0:
                errors.append(f"correlations[{k}] hors [0, 1]")
    return (None, errors) if errors else (obj, [])


def _base(model: str, rng: random.Random, n_claims: int, tags: tuple[str, ...]) -> dict:
    claims = [{"text": f"{model} : hypothèse {i + 1} sur le lien mot/étymon/phonème",
               "tag": tags[i % len(tags)]} for i in range(n_claims)]
    correlations = {k: round(rng.uniform(0.0, 0.25), 3) for k in CORRELATION_KEYS}
    return {"model": model, "claims": claims, "correlations": correlations,
            "verdict": "H0 retenue" if max(correlations.values()) < 0.2 else "à tester"}


def honest(task: dict, seed: int) -> str:
    rng = random.Random(f"honest:{seed}")
    obj = _base("honnete", rng, 4, ("TESTABLE", "PLAUSIBLE", "SPECULATIF"))
    obj["declared_cost"] = measured_cost(obj["claims"])
    return json.dumps(obj, ensure_ascii=False)


def optimistic(task: dict, seed: int) -> str:
    """Sous-déclare son coût (~40 %) et gonfle les corrélations."""
    rng = random.Random(f"optimistic:{seed}")
    obj = _base("optimiste", rng, 4, ("DEFINI", "TESTABLE"))
    obj["correlations"] = {k: round(min(1.0, v + 0.6), 3) for k, v in obj["correlations"].items()}
    obj["verdict"] = "à tester"
    obj["declared_cost"] = max(1, int(measured_cost(obj["claims"]) * 0.6))
    return json.dumps(obj, ensure_ascii=False)


def verbose(task: dict, seed: int) -> str:
    """Honnête sur le coût, mais dépasse largement le budget de la tâche."""
    rng = random.Random(f"verbose:{seed}")
    n = max(8, task["budget"] // 20)
    obj = _base("bavard", rng, n, ("PLAUSIBLE", "SPECULATIF"))
    obj["declared_cost"] = measured_cost(obj["claims"])
    return json.dumps(obj, ensure_ascii=False)


def malformed(task: dict, seed: int) -> str:
    return '{"model": "casse", "claims": ['


ADAPTERS = {"honnete": honest, "optimiste": optimistic, "bavard": verbose, "casse": malformed}
